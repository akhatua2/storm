import json
import logging
from typing import Dict, List, Any, Callable, Optional, Union
from .utils import extract_json_from_response, format_knowledge_graph_for_prompt, optimize_knowledge_graph

class InconsistencyDetector:
    """
    A class that analyzes search results to build a knowledge graph and detect inconsistencies
    using an LLM for analysis.
    """
    
    def __init__(self, llm: Callable, topic: Optional[str] = None):
        """
        Initialize the InconsistencyDetector.
        
        Args:
            llm: A callable that takes a prompt and returns a response
            topic: Optional topic to focus the analysis on
        """
        self.topic = topic
        self.llm = llm
        
        # Initialize knowledge graph with empty facts and inconsistencies
        self.knowledge_graph = {
            "facts": [],  # List of verified facts with their sources
            "inconsistencies": []  # List of inconsistencies with their sources and reasoning
        }
        
        # Track processed URLs to avoid duplicates
        self.processed_urls = set()
        
        # Extract initial facts from the topic if provided
        if self.topic:
            self._extract_initial_facts()
    
    def _extract_initial_facts(self) -> None:
        """
        Extract initial facts about the topic to initialize the knowledge graph.
        """
        try:
            prompt = f"""
            Extract factual information explicitly stated in this specific sentence: "{self.topic}".
            
            Your task is to identify only the facts that are directly stated in the input sentence.
            Do not include any information from external knowledge or inference.
            Return each fact as a separate item in the list.
            
            Example:
            Input: "John Smith is a 45-year-old American doctor who graduated from Harvard in 2001."
            Output: ["John Smith is 45 years old", "John Smith is American", "John Smith is a doctor", "John Smith graduated from Harvard in 2001"]
            
            Format your response as a JSON array of strings, like this:
            ["fact 1", "fact 2", "fact 3"]
            """
            
            # Get response from LLM
            response = self.llm(prompt)
            
            # Handle list response (some LLMs return a list with one element)
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            
            # Process the response
            try:
                # Extract JSON content from response
                cleaned_response = extract_json_from_response(response)
                
                # Parse the JSON response
                facts = json.loads(cleaned_response)
                
                # Ensure facts is a list
                if isinstance(facts, list):
                    for fact in facts:
                        self.knowledge_graph["facts"].append(fact)
                else:
                    logging.error(f"Unexpected facts format: {facts}")
                    
                print("-------------knowledge_graph-------------------")
                print(self.knowledge_graph)
                print("--------------------------------")
            except json.JSONDecodeError:
                logging.error(f"Failed to parse response as JSON: {response}")
        except Exception as e:
            logging.error(f"Error extracting initial facts: {str(e)}")
            logging.exception(e)
        
    def process_search_results(self, search_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process search results from BingSearch to build knowledge graph and detect inconsistencies.
        
        Args:
            search_results: List of dictionaries from BingSearch results
            
        Returns:
            Dictionary with 'facts' and 'inconsistencies' lists
        """
        for result in search_results:
            url = result.get("url", "")
            if url in self.processed_urls:
                continue
                
            self.processed_urls.add(url)
            
            # Extract snippets from the result
            snippets = result.get("snippets", [])
            title = result.get("title", "")
            description = result.get("description", "")
            
            # Process each snippet
            for snippet in snippets:
                self._process_chunk(snippet, url, title, description)
                
        print("------------knowledge_graph--------------------")
        print(self.knowledge_graph["inconsistencies"])
        print("-----------------------------------------------")
                
        return {
            "facts": self.knowledge_graph["facts"],
            "inconsistencies": self.knowledge_graph["inconsistencies"]
        }
    
    def _process_chunk(self, chunk: str, url: str, title: str, description: str) -> None:
        """
        Process a single chunk of text to extract facts and detect inconsistencies.
        
        Args:
            chunk: Text chunk to analyze
            url: Source URL
            title: Title of the source
            description: Description of the source
        """
        # Create context from current knowledge graph
        context = format_knowledge_graph_for_prompt(self.knowledge_graph)
        # print("--------------context------------------")
        # print(context)
        # print("---------------------------------------")
        
        # Create prompt for verification
        prompt = f"""
        Verify new information against existing knowledge and detect inconsistencies.
        
        Topic: {self.topic if self.topic else "the topic"}
        
        Current knowledge:
        {context}
        
        New information from {url}:
        {chunk}
        
        Your task is to analyze new information against existing knowledge to identify:
        1. New facts that are not already known about the MAIN ENTITY or topic
        2. ONLY ACTUAL inconsistencies between the new information and existing knowledge
        
        Types of inconsistencies to look for:
        - Contradictory facts (e.g., "X was born in 1980" vs "X was born in 1985")
        - Conflicting attributes (e.g., "X is a Republican" vs "X is a Democrat")
        - Temporal inconsistencies (e.g., "X died in 1990" vs "X was active in 2000")
        - Numerical contradictions (e.g., "X has 3 children" vs "X has 5 children")
        - Causal inconsistencies (e.g., "X caused Y" vs "Y caused X")
        - Identity confusion (e.g., "X is the CEO" vs "Y is the CEO")
        - Status contradictions (e.g., "X is alive" vs "X is deceased")
        - Relationship inconsistencies (e.g., "X is married to Y" vs "X is married to Z")
        
        CRITICAL GUIDELINES - FOLLOW THESE EXACTLY:
        1. NEVER flag omissions as inconsistencies. The absence of information is NOT a contradiction.
        2. New publications, research interests, or activities are NOT inconsistencies unless they directly contradict previous information.
        3. Each inconsistency should be reported only ONCE with the clearest example of contradiction.
        4. Be precise - point to specific statements that contradict each other.
        5. Verify that each reported inconsistency involves a genuine logical contradiction between statements, not just new information.
        6. If you find yourself writing "not necessarily a contradiction" or "there is no explicit contradiction" in your reasoning, then DO NOT include it as an inconsistency.
        7. Only include an item in the inconsistencies list if you are 100% certain it represents a direct logical contradiction.
        8. Different perspectives, additional details, or complementary information are NOT inconsistencies.
        9. DO NOT include general background information or contextual facts about the country, industry, or field unless they directly relate to the main entity.
        10. DO NOT DUPLICATE inconsistencies - each specific contradiction should appear exactly once in your response.
        11. Focus strictly on the main entity or central topic - not peripheral information about related entities or general context.
        
        Format your response as a JSON object with the following structure:
        {{
            "new_facts": ["fact 1", "fact 2", ...],
            "inconsistencies": [
                {{"description": "description of inconsistency 1", "reasoning": "reasoning for inconsistency 1"}},
                {{"description": "description of inconsistency 2", "reasoning": "reasoning for inconsistency 2"}},
                ...
            ]
        }}
        
        DO NOT include facts and inconsistencies that are already in the knowledge graph.
        Leave the inconsistencies list empty if you don't find any TRUE contradictions.
        """
        
        # print("--------------prompt------------------")
        # print(prompt)
        # print("--------------------------------")
        
        try:
            # Get response from LLM
            response = self.llm(prompt)
            # print("--------------response------------------")
            # print(response)
            # print("--------------------------------")
            
            # Handle list response (some LLMs return a list with one element)
            if isinstance(response, list) and len(response) > 0:
                response = response[0]
            
            # Process the response
            try:
                # Extract JSON content from response
                cleaned_response = extract_json_from_response(response)
                
                # Parse the JSON response
                result = json.loads(cleaned_response)
                
                # Add new facts to knowledge graph
                new_facts = result.get("new_facts", [])
                if isinstance(new_facts, list):
                    for fact in new_facts:
                        # self.knowledge_graph["facts"].append({
                        #     "fact": fact,
                        #     "source": url,
                        #     "title": title
                        # })
                        self.knowledge_graph["facts"].append(fact)
                else:
                    logging.error(f"Unexpected new_facts format: {new_facts}")
                
                # Add inconsistencies to knowledge graph
                inconsistencies = result.get("inconsistencies", [])
                if isinstance(inconsistencies, list):
                    # Get existing descriptions to avoid duplicates
                    existing_descriptions = {inc["description"] for inc in self.knowledge_graph["inconsistencies"] if "description" in inc}
                    
                    for inconsistency in inconsistencies:
                        if isinstance(inconsistency, dict):
                            description = inconsistency.get("description", "")
                            # Only add if this description doesn't already exist
                            if description and description not in existing_descriptions:
                                existing_descriptions.add(description)
                                self.knowledge_graph["inconsistencies"].append({
                                    "description": description,
                                    "reasoning": inconsistency.get("reasoning", ""),
                                    # "source": url,
                                    # "title": title
                                })
                        else:
                            # Handle case where inconsistency is a string
                            description = str(inconsistency)
                            if description and description not in existing_descriptions:
                                existing_descriptions.add(description)
                                self.knowledge_graph["inconsistencies"].append({
                                    "description": description,
                                    "reasoning": "",
                                    # "source": url,
                                    # "title": title
                                })
                else:
                    logging.error(f"Unexpected inconsistencies format: {inconsistencies}")
                    
                # Optimize knowledge graph if we have too many facts
                if len(self.knowledge_graph["facts"]) > 50:
                    self.knowledge_graph = optimize_knowledge_graph(self.knowledge_graph, self.llm, topic=self.topic)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse response as JSON: {response}")
        except Exception as e:
            logging.error(f"Error processing chunk from {url}: {str(e)}")
            logging.exception(e)
    
    def get_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the current facts and inconsistencies.
        
        Returns:
            Dictionary with 'facts' and 'inconsistencies' lists
        """
        return {
            "facts": self.knowledge_graph["facts"],
            "inconsistencies": self.knowledge_graph["inconsistencies"]
        }
