import logging
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import re

from .storm_wiki.modules.storm_dataclass import StormInformationTable

class CorpusLevelInconsistencyResolver:
    """
    A class that resolves inconsistencies at the corpus level by analyzing the relevant context
    for each inconsistency and making a determination based on the evidence.
    """
    
    def __init__(self, llm: Callable):
        """
        Initialize the CorpusLevelInconsistencyResolver.
        
        Args:
            llm: A callable that takes a prompt and returns a response
        """
        self.llm = llm
    
    def resolve_inconsistencies(self, 
                               information_table: StormInformationTable, 
                               inconsistencies: List[Dict[str, str]],
                               top_k: int = 5) -> StormInformationTable:
        """
        Resolve inconsistencies by analyzing relevant chunks from the information table.
        
        Args:
            information_table: The StormInformationTable containing all collected information
            inconsistencies: List of inconsistency dictionaries with 'description' and 'reasoning' fields
            top_k: Number of most relevant chunks to retrieve for each inconsistency
            
        Returns:
            Updated StormInformationTable with resolved/removed inconsistencies
        """
        if not inconsistencies:
            logging.info("No inconsistencies to resolve")
            return information_table
            
        logging.info(f"Starting resolution of {len(inconsistencies)} inconsistencies with top_k={top_k}")
            
        # Make sure information table is prepared for retrieval
        if not hasattr(information_table, 'encoded_snippets') or information_table.encoded_snippets is None:
            logging.info("Preparing information table for retrieval")
            information_table.prepare_table_for_retrieval()
        
        # Process each inconsistency
        resolved_inconsistencies = []
        total_facts_removed = 0
        total_facts_added = 0
        total_snippets_corrected = 0
        
        # Track which snippets we've already corrected to avoid duplicating work
        corrected_snippet_indices = set()
        
        for i, inconsistency in enumerate(inconsistencies):
            description = inconsistency.get('description', '')
            reasoning = inconsistency.get('reasoning', '')
            
            if not description:
                logging.warning(f"Skipping inconsistency #{i} as it has no description")
                continue
                
            logging.info(f"Processing inconsistency #{i+1}/{len(inconsistencies)}: {description[:100]}..." if len(description) > 100 else description)
                
            # Retrieve relevant chunks for this inconsistency
            relevant_chunks, chunk_urls, chunk_indices = self._retrieve_relevant_chunks(
                information_table, description, top_k
            )
            
            logging.info(f"Found {len(relevant_chunks)} relevant chunks for inconsistency #{i+1}")
            for j, (chunk, url) in enumerate(zip(relevant_chunks, chunk_urls)):
                logging.debug(f"Relevant chunk #{j+1} from {url}: {chunk[:100]}..." if len(chunk) > 100 else chunk)
            
            # Check if we got enough relevant information
            if not relevant_chunks or len(relevant_chunks) < 2:
                # Not enough information to resolve, keep the inconsistency
                logging.info(f"Not enough relevant information to resolve inconsistency #{i+1}, keeping it")
                resolved_inconsistencies.append(inconsistency)
                continue
                
            # Determine resolution using LLM
            logging.info(f"Asking LLM to resolve inconsistency #{i+1}")
            resolution_result = self._resolve_with_llm(description, reasoning, relevant_chunks, chunk_urls)
            logging.info(f"LLM resolution type: {resolution_result['resolution_type']}")
            
            if resolution_result['resolution_type'] == 'keep':
                # Keep the inconsistency, it's a genuine unresolvable issue
                logging.info(f"Keeping inconsistency #{i+1} as it's genuinely unresolvable")
                resolved_inconsistencies.append(inconsistency)
                
            elif resolution_result['resolution_type'] == 'remove':
                # This inconsistency should be removed (e.g., refers to unrelated entities)
                logging.info(f"Removing inconsistency #{i+1}: {description} - {resolution_result['explanation']}")
                
            elif resolution_result['resolution_type'] == 'resolved':
                # The inconsistency was resolved, now we need to fix the information in the table
                if 'correct_fact' in resolution_result and resolution_result['correct_fact']:
                    # Get the URLs of incorrect information
                    incorrect_urls = resolution_result.get('incorrect_urls', [])
                    if not incorrect_urls and 'incorrect_url' in resolution_result:
                        incorrect_urls = [resolution_result['incorrect_url']]
                    
                    # Get source URL for the correct fact
                    source_url = resolution_result.get('source_url', '')
                    correct_fact = resolution_result['correct_fact']
                    
                    logging.info(f"Resolved inconsistency #{i+1}: {description}")
                    logging.info(f"Correct fact: '{correct_fact}' from source: {source_url}")
                    logging.info(f"Incorrect information in URLs: {incorrect_urls}")
                    
                    # First, add the correct fact to ALL snippets that will be retrieved for this query
                    # This ensures that even when retrieved via vector search, the correct information is shown
                    for index, snippet in enumerate(information_table.collected_snippets):
                        if self._snippet_contains_inconsistency(snippet, description):
                            if index not in corrected_snippet_indices:
                                corrected_snippet = self._correct_snippet(snippet, description, correct_fact)
                                information_table.collected_snippets[index] = corrected_snippet
                                total_snippets_corrected += 1
                                corrected_snippet_indices.add(index)
                                logging.info(f"Corrected snippet #{index}: {snippet[:100]}..." if len(snippet) > 100 else snippet)
                    
                    # Also update the Information objects' snippets for the incorrect URLs
                    for url in incorrect_urls:
                        if url in information_table.url_to_info:
                            logging.info(f"Processing snippets in information object for URL: {url}")
                            info_obj = information_table.url_to_info[url]
                            
                            # For each snippet in this Information object
                            corrected_snippets = []
                            for snippet in info_obj.snippets:
                                if self._snippet_contains_inconsistency(snippet, description):
                                    corrected_snippet = self._correct_snippet(snippet, description, correct_fact)
                                    corrected_snippets.append(corrected_snippet)
                                    logging.info(f"Corrected snippet in URL {url}: {snippet[:100]}..." if len(snippet) > 100 else snippet)
                                else:
                                    corrected_snippets.append(snippet)
                            
                            # Replace with corrected snippets
                            info_obj.snippets = corrected_snippets
                            logging.info(f"Updated snippets in Information object for URL: {url}")
                    
                    # Update facts in the Information objects
                    if source_url and source_url in information_table.url_to_info:
                        # Add the corrected fact if it's not already there
                        info_obj = information_table.url_to_info[source_url]
                        if not hasattr(info_obj, 'facts'):
                            info_obj.facts = []
                        
                        if correct_fact not in info_obj.facts:
                            info_obj.facts.append(correct_fact)
                            total_facts_added += 1
                            logging.info(f"Added correct fact to {source_url}: '{correct_fact}'")
                        else:
                            logging.info(f"Correct fact already exists in {source_url}")
                    
                    # Remove incorrect facts from incorrect sources
                    for url in incorrect_urls:
                        if url in information_table.url_to_info:
                            # Find and remove incorrect facts
                            info_obj = information_table.url_to_info[url]
                            if hasattr(info_obj, 'facts') and info_obj.facts:
                                facts_before = len(info_obj.facts)
                                facts_to_keep = []
                                for fact in info_obj.facts:
                                    # Check if this fact relates to the inconsistency
                                    if not self._fact_relates_to_inconsistency(fact, description):
                                        facts_to_keep.append(fact)
                                    else:
                                        logging.info(f"Removing incorrect fact from {url}: '{fact}'")
                                        total_facts_removed += 1
                                
                                # Update with only the facts to keep
                                info_obj.facts = facts_to_keep
                                logging.info(f"Removed {facts_before - len(facts_to_keep)} facts from {url}")
        
        # After correcting snippets, we need to re-encode them
        logging.info(f"Total snippets corrected: {total_snippets_corrected}")
        if total_snippets_corrected > 0:
            logging.info("Re-encoding corrected snippets")
            information_table.encoded_snippets = information_table.encoder.encode(information_table.collected_snippets)
        
        # Update the knowledge graph in the information table
        # Collect all facts from url_to_info entries
        old_fact_count = len(information_table.knowledge_graph.get("facts", [])) if hasattr(information_table, "knowledge_graph") else 0
        old_inconsistency_count = len(information_table.knowledge_graph.get("inconsistencies", [])) if hasattr(information_table, "knowledge_graph") else 0
        
        all_facts = []
        for url, info_obj in information_table.url_to_info.items():
            if hasattr(info_obj, 'facts') and info_obj.facts:
                all_facts.extend(info_obj.facts)
        
        information_table.knowledge_graph = {
            "facts": all_facts,
            "inconsistencies": resolved_inconsistencies
        }
        
        # Log the summary of changes
        logging.info(f"Inconsistency resolution complete:")
        logging.info(f"- Facts: {old_fact_count} → {len(all_facts)} (removed: {total_facts_removed}, added: {total_facts_added})")
        logging.info(f"- Inconsistencies: {old_inconsistency_count} → {len(resolved_inconsistencies)}")
        logging.info(f"- Snippets corrected: {total_snippets_corrected}")
        
        return information_table
    
    def _fact_relates_to_inconsistency(self, fact: str, inconsistency_description: str) -> bool:
        """
        Check if a fact relates to a given inconsistency description.
        
        Args:
            fact: The fact to check
            inconsistency_description: Description of the inconsistency
            
        Returns:
            True if the fact is related to the inconsistency, False otherwise
        """
        # Simple string matching approach - could be enhanced with semantic similarity
        fact_lower = fact.lower()
        desc_lower = inconsistency_description.lower()
        
        # Extract key entities and values from the inconsistency description
        key_terms = self._extract_key_terms(desc_lower)
        
        # Check if enough key terms are in the fact
        matches = sum(1 for term in key_terms if term in fact_lower)
        return matches >= min(2, len(key_terms))
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text for matching.
        
        Args:
            text: The text to extract terms from
            
        Returns:
            List of key terms
        """
        # Split by common separators and filter out short terms
        terms = []
        for sep in [' vs ', ' and ', ' or ', ': ', ', ', '. ', ' states ', ' claims ']:
            if sep in text:
                parts = text.split(sep)
                terms.extend([p.strip() for p in parts if len(p.strip()) > 3])
        
        # If no terms found with separators, use words
        if not terms:
            terms = [word for word in text.split() if len(word) > 3]
            
        return list(set(terms))
    
    def _snippet_contains_inconsistency(self, snippet: str, inconsistency_description: str) -> bool:
        """
        Check if a snippet contains the inconsistent information.
        
        Args:
            snippet: The snippet to check
            inconsistency_description: Description of the inconsistency
            
        Returns:
            True if the snippet contains the inconsistency, False otherwise
        """
        # More sophisticated approach for matching snippets with inconsistencies
        snippet_lower = snippet.lower()
        desc_lower = inconsistency_description.lower()
        
        # Extract key numbers from the inconsistency description - useful for numerical inconsistencies
        numbers = re.findall(r'\d+', desc_lower)
        if numbers:
            # If inconsistency mentions numbers, check if any of these appear in the snippet
            for num in numbers:
                if num in snippet_lower:
                    return True
        
        # Extract key terms for semantic matching
        key_terms = self._extract_key_terms(desc_lower)
        
        # Check for more specific terms that indicate this kind of inconsistency
        specific_indicators = []
        if "citation" in desc_lower or "cited" in desc_lower:
            specific_indicators.extend(["citation", "cited", "cites", "h-index", "impact"])
        if "article" in desc_lower or "publication" in desc_lower:
            specific_indicators.extend(["article", "publication", "published", "paper"])
        if "date" in desc_lower or "year" in desc_lower or "born" in desc_lower:
            specific_indicators.extend(["date", "year", "born", "birthday", "birth"])
        
        # Check for key terms and specific indicators
        term_matches = sum(1 for term in key_terms if term in snippet_lower)
        indicator_matches = sum(1 for indicator in specific_indicators if indicator in snippet_lower)
        
        # More aggressive matching for snippets - we want to catch more potential matches
        return term_matches >= min(2, len(key_terms)) or indicator_matches > 0
    
    def _correct_snippet(self, snippet: str, inconsistency_description: str, correct_fact: str) -> str:
        """
        Add a correction note to a snippet containing incorrect information.
        
        Args:
            snippet: The snippet to correct
            inconsistency_description: Description of the inconsistency
            correct_fact: The correct fact
            
        Returns:
            Corrected snippet
        """
        # Add a visible correction that will stand out in the final output
        if not snippet.endswith('\n'):
            snippet += '\n'
        
        # Make the correction extremely visible with clear format that will appear in final output
        correction = f"\n\n=================================================\n"
        correction += f"IMPORTANT CORRECTION: {correct_fact}\n"
        correction += f"=================================================\n\n"
        
        # Try to identify specific incorrect information in the snippet for clearer correction
        numbers_in_snippet = re.findall(r'\d+', snippet)
        numbers_in_fact = re.findall(r'\d+', correct_fact)
        
        if numbers_in_snippet and numbers_in_fact:
            # If both have numbers, try to highlight what was incorrect
            correction += f"NOTE: The information above contains errors. "
            correction += f"The correct information is: {correct_fact}\n\n"
        
        # Find where to insert the correction - we want it near the incorrect information
        # but not breaking the flow of text too much
        lines = snippet.split('\n')
        
        # Try to find the best position for insertion - right after paragraph with incorrect info
        for i, line in enumerate(lines):
            if self._line_contains_inconsistency(line, inconsistency_description):
                # Insert correction after this line
                lines.insert(i+1, correction)
                return '\n'.join(lines)
        
        # If we can't find a good position, just append it to the end
        return snippet + correction
    
    def _line_contains_inconsistency(self, line: str, inconsistency_description: str) -> bool:
        """Check if a specific line contains the inconsistency."""
        line_lower = line.lower()
        desc_lower = inconsistency_description.lower()
        
        # Extract numbers from the inconsistency description
        numbers = re.findall(r'\d+', desc_lower)
        if numbers:
            for num in numbers:
                if num in line_lower:
                    return True
        
        # Try key terms match
        key_terms = self._extract_key_terms(desc_lower)
        term_matches = sum(1 for term in key_terms if term in line_lower)
        return term_matches >= min(2, len(key_terms))
    
    def _retrieve_relevant_chunks(self, 
                                 information_table: StormInformationTable, 
                                 query: str,
                                 top_k: int = 5) -> Tuple[List[str], List[str], List[int]]:
        """
        Retrieve the most relevant chunks for a given inconsistency.
        
        Args:
            information_table: The StormInformationTable containing all collected information
            query: The inconsistency description to search for
            top_k: Number of most relevant chunks to retrieve
            
        Returns:
            Tuple of (relevant chunks, their source URLs, indices in the original list)
        """
        # Get information using the built-in retrieve_information method
        information_objects = information_table.retrieve_information(query, top_k)
        
        relevant_chunks = []
        chunk_urls = []
        chunk_indices = []
        
        # Find the indices of these chunks in the original list
        for info in information_objects:
            for snippet in info.snippets:
                try:
                    # Find the index of this snippet in the collected_snippets
                    index = information_table.collected_snippets.index(snippet)
                    relevant_chunks.append(snippet)
                    chunk_urls.append(info.url)
                    chunk_indices.append(index)
                except ValueError:
                    # Snippet not found in collected_snippets, add it without an index
                    relevant_chunks.append(snippet)
                    chunk_urls.append(info.url)
                    chunk_indices.append(-1)
        
        return relevant_chunks, chunk_urls, chunk_indices
    
    def _resolve_with_llm(self, 
                         description: str, 
                         reasoning: str,
                         relevant_chunks: List[str], 
                         chunk_urls: List[str]) -> Dict[str, Any]:
        """
        Use LLM to resolve the inconsistency based on relevant chunks.
        
        Args:
            description: The inconsistency description
            reasoning: The reasoning behind the inconsistency
            relevant_chunks: List of relevant text chunks
            chunk_urls: List of source URLs for the chunks
            
        Returns:
            Dictionary with resolution details:
            - resolution_type: 'keep', 'remove', or 'resolved'
            - explanation: Explanation of the resolution
            - correct_fact: The correct fact if resolved
            - source_url: Source URL of the correct fact
            - incorrect_urls: List of URLs containing incorrect information
        """
        # Create context from the relevant chunks
        context = "\n\n".join([f"Source {i+1} ({url}):\n{chunk}" 
                              for i, (chunk, url) in enumerate(zip(relevant_chunks, chunk_urls))])
        
        prompt = f"""
        I need your help resolving an inconsistency in the information collected about a topic.
        
        INCONSISTENCY:
        {description}
        
        REASONING:
        {reasoning}
        
        RELEVANT INFORMATION FROM SOURCES:
        {context}
        
        Please analyze the relevant information and determine:
        1. If the inconsistency refers to unrelated entities (e.g., two different people with the same name)
        2. If one of the contradicting claims is clearly correct based on the evidence
        3. If the inconsistency is genuine and unresolvable based on the available information
        
        Choose one of the following resolution types:
        - "remove": If the inconsistency refers to unrelated entities or is based on a misunderstanding
        - "resolved": If you can determine which claim is correct
        - "keep": If the inconsistency is genuine and unresolvable with the available information
        
        Format your response as a JSON object with the following structure:
        {{
            "resolution_type": "remove" | "resolved" | "keep",
            "explanation": "detailed explanation of your reasoning",
            "correct_fact": "the correct fact statement (only if resolution_type is 'resolved')",
            "source_url": "the URL of the most reliable source (only if resolution_type is 'resolved')",
            "incorrect_urls": ["list", "of", "URLs", "with", "incorrect", "information"]
        }}
        
        IMPORTANT: When resolution_type is "resolved", you MUST identify both the correct source URL and the URLs that contain incorrect information. This is crucial for properly fixing the inconsistency.
        """
        
        try:
            # Get response from LLM
            response = self.llm(prompt)
            
            # Handle list response (some LLMs return a list with one element)
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            
            # Process the response
            from .utils import extract_json_from_response
            import json
            
            # Extract JSON content from response
            cleaned_response = extract_json_from_response(response)
            
            # Parse the JSON response
            result = json.loads(cleaned_response)
            
            # Add default fields if missing
            if 'resolution_type' not in result:
                result['resolution_type'] = 'keep'
            if 'explanation' not in result:
                result['explanation'] = 'Unable to determine resolution'
            if 'incorrect_urls' not in result:
                result['incorrect_urls'] = []
                
            return result
            
        except Exception as e:
            logging.error(f"Error in LLM resolution: {str(e)}")
            return {
                'resolution_type': 'keep',
                'explanation': f'Error in resolution process: {str(e)}',
                'incorrect_urls': []
            }
