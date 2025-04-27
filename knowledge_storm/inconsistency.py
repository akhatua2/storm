import json
import logging
from typing import Dict, List, Any, Callable, Optional, Union
from .utils import extract_json_from_response, format_knowledge_graph_for_prompt, optimize_knowledge_graph

class InconsistencyDetector:
    """
    A class that analyzes search results to build a knowledge graph and detect unique entity identifiers.
    This helps distinguish between different entities that might share the same name.
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
        
        # Initialize knowledge graph with triplets
        self.knowledge_graph = {
            "triplets": [],  # List of structured triplets [Subject, Predicate, Object]
        }
        
        # Track processed URLs to avoid duplicates
        self.processed_urls = set()
        
        # Track accepted and rejected URLs
        self.accepted_urls = set()  # URLs that refer to the same entity or relevant context
        self.rejected_urls = set()  # URLs that refer to completely different entities
        self.url_rejection_reasons = {}  # Store reasons for rejection
        
        # Extract initial identifiers from the topic if provided
        if self.topic:
            self._extract_initial_identifiers()
    
    def _extract_initial_identifiers(self) -> None:
        """
        Extract initial identifying triplets about the topic to initialize the knowledge graph.
        """
        try:
            prompt = f"""
            Extract ONLY specific unique knowledge graph triplets for "{self.topic}" that would definitively distinguish this exact entity from any other entity with a similar name or purpose. Represent the information as structured triplets: [Subject, Predicate, Object].

            Context: The Movement for the Liberation of the Congo (MLC) is a political militia and political party operating in the Democratic Republic of Congo (DRC). Founded during the Second Congo War, the MLC initially functioned as a rebel group and later transitioned into a political organization. It is led by Jean-Pierre Bemba, a prominent Congolese politician and former rebel leader. The MLC is part of the Union Sacrée political platform, supporting President Félix Tshisekedi, but has expressed dissatisfaction with its political standing and electoral outcomes. The group has been involved in protests and demonstrations, often advocating for its political interests and seeking greater recognition for its contributions within the national political landscape.

            CRITICAL INSTRUCTIONS:
            - Focus EXCLUSIVELY on triplets that are SPECIFIC to {self.topic} AS AN ORGANIZATION.
            - These triplets should function like a fingerprint - unique combinations that could only apply to this specific entity.
            - Extract relationships such as:
                * Founding Details: [{self.topic}, "founded on", "Date"], [{self.topic}, "founded in", "Location"]
                * Leadership: [{self.topic}, "led by", "Leader Name"], ["Leader Name", "role is", "Leader Role"]
                * Location: [{self.topic}, "headquartered in", "Location"]
                * Mission/Mandate: [{self.topic}, "mission is", "Mission Statement Snippet"]
                * Key Actions: [{self.topic}, "organized", "Event Name"], ["Event Name", "occurred on", "Date"]
                * Affiliations/Engagements: [{self.topic}, "engaged with", "Other Entity"], [{self.topic}, "affiliated with", "Parent Organization"]
                * Legal Status: [{self.topic}, "legal status is", "Status Description"]
            - Ensure Subject, Predicate, and Object are concise strings.

            DO NOT EXTRACT:
            - General facts about the sector, industry, or field this entity operates in.
            - Statistics about the broader category of entities.
            - Policy information unless specifically created or championed by this entity.
            - Background context unless uniquely tied to this entity's origin.
            - Generic challenges faced by similar entities.

            Format your response as a JSON array of triplets (lists of strings):
            [["Subject1", "Predicate1", "Object1"], ["Subject2", "Predicate2", "Object2"], ...]
            Example:
            [
                ["MLC", "led by", "Jean-Pierre Bemba"],
                ["Jean-Pierre Bemba", "is a", "prominent Congolese politician"],
                ["MLC", "operates in", "Democratic Republic of Congo"],
                ["MLC", "founded during", "Second Congo War"]
            ]
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
                triplets = json.loads(cleaned_response)
                
                # Ensure triplets is a list of lists (triplets)
                if isinstance(triplets, list) and all(isinstance(t, list) and len(t) == 3 for t in triplets):
                    self.knowledge_graph["triplets"].extend(triplets)
                else:
                    logging.error(f"Unexpected triplets format: {triplets}")
                    
                print("-------------knowledge_graph (triplets)-------------------")
                print(self.knowledge_graph["triplets"])
                print("-----------------------------------------------")
            except json.JSONDecodeError:
                logging.error(f"Failed to parse response as JSON: {response}")
        except Exception as e:
            logging.error(f"Error extracting initial identifiers: {str(e)}")
            logging.exception(e)
        
    def process_search_results(self, search_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process search results to build knowledge graph and identify different entities.
        
        Args:
            search_results: List of dictionaries from search results
            
        Returns:
            Dictionary with 'triplets', 'accepted_urls', and 'rejected_urls'
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
            
            # Initialize as accepting this URL by default - reject only if we confirm it's irrelevant
            is_valid_url = True
            rejection_reason = ""
            
            # Process each snippet
            for snippet in snippets:
                snippet_result = self._process_chunk(snippet, url, title, description)
                
                # If any snippet indicates the content should be rejected, reject the URL
                if snippet_result and not snippet_result.get("should_accept", True):
                    is_valid_url = False
                    rejection_reason = snippet_result.get("reasoning", "Content not relevant for article writing")
                    break
            
            # Track this URL as accepted or rejected
            if is_valid_url:
                self.accepted_urls.add(url)
                logging.info(f"Accepted URL as relevant to topic: {url}")
            else:
                self.rejected_urls.add(url)
                self.url_rejection_reasons[url] = rejection_reason
                logging.info(f"Rejected URL as irrelevant: {url} - Reason: {rejection_reason}")
                
        print("------------knowledge_graph (triplets)--------------------")
        print(self.knowledge_graph["triplets"])
        print("-----------------------------------------------")
        
        # Include accepted and rejected URLs in the return value
        return {
            "triplets": self.knowledge_graph["triplets"], # Return triplets instead of facts
            "accepted_urls": list(self.accepted_urls),
            "rejected_urls": list(self.rejected_urls)
        }
    
    def _process_chunk(self, chunk: str, url: str, title: str, description: str) -> Optional[Dict[str, Any]]:
        """
        Process a single chunk of text to extract entity identifiers and detect entity disambiguation issues.
        
        Args:
            chunk: Text chunk to analyze
            url: Source URL
            title: Title of the source
            description: Description of the source
            
        Returns:
            Dictionary with processing results or None if an error occurred
        """
        # Create context from current knowledge graph
        # NOTE: format_knowledge_graph_for_prompt needs to be updated to handle triplets
        context = format_knowledge_graph_for_prompt(self.knowledge_graph) 
        
        # Create prompt for verification
        prompt = f"""
        Analyze whether this information is useful for writing a comprehensive article about "{self.topic if self.topic else "the topic"}" or if it should be rejected completely.

        Current knowledge graph triplets for {self.topic}:
        {context} 

        New information:
        Title: {title}
        Description: {description}
        Content: {chunk}

        ARTICLE WRITING TASK:
        We are writing a comprehensive article about {self.topic} and need to gather ALL relevant information, including broader context, history, regulations, and related developments that would help readers understand the full picture.

        ACCEPTANCE CRITERIA (BE INCLUSIVE - Accept if ANY of these apply):
        1. Information directly about {self.topic}
        2. Information about the broader context, field, or domain in which {self.topic} operates (e.g., gig economy, labor rights)
        3. Information about similar organizations in the same field or with similar goals
        4. Information about policies, regulations, or legislation relevant to {self.topic}'s mission, even if not explicitly connected
        5. Statistical data or background that helps understand the issues {self.topic} addresses
        6. Information about related events, protests, or advocacy work in the same domain
        7. Historical context that helps explain the emergence or importance of organizations like {self.topic}
        8. Economic, social, or technological trends that impact the environment in which {self.topic} operates

        SPECIFIC EXAMPLES OF WHAT TO ACCEPT:
        - Comparative analyses between gig economy and traditional employment
        - Budget announcements affecting gig workers or labor rights
        - Historical evolution of labor movements or gig work
        - Information about other gig economy platforms or labor organizations
        - Government reports or statistics about the gig economy or worker conditions

        REJECTION CRITERIA (BE VERY STRICT - ONLY reject if ALL THREE of these apply):
        1. The content is clearly about a COMPLETELY DIFFERENT ENTITY that just happens to share a similar name or acronym
        2. The content focuses on a completely different field or sector with NO connection to {self.topic}'s domain
        3. The content has ABSOLUTELY NO RELEVANCE to the social, political, or economic context in which {self.topic} operates

        DISAMBIGUATION NOTE:
        - Even if content discusses a different organization with a similar name or acronym, ACCEPT it if that organization operates in the same field or addresses similar issues
        - For example, reject "Impact Guru Foundation (IGF)" when writing about "Indian Gig Workers Front (IGF)" ONLY if the content has no relevance to gig workers or labor issues
        - When in doubt, ACCEPT the content - it's better to include potentially relevant information than to exclude it

        INFORMATION ASSESSMENT & TRIPLET EXTRACTION:
        - If accepting information directly about {self.topic}, extract specific new knowledge graph triplets related to the entity in the format ["Subject", "Predicate", "Object"].
        - Ensure Subject, Predicate, and Object are concise strings.
        - Focus on extracting the relationships described in the initial extraction prompt (Leadership, Key Actions, Affiliations, etc.).
        - If accepting contextual information (criteria 2-8), return an empty list for new_triplets.

        Output as a JSON object:
        {{
            "should_accept": true/false, // Use false ONLY for completely different entities with no contextual relevance
            "reasoning": "Detailed explanation of why this information should be accepted or rejected",
            "relevance_category": "direct_information/contextual_background/related_organization/policy_context/historical_context/etc.",
            "new_triplets": [["Subject1", "Predicate1", "Object1"], ["Subject2", "Predicate2", "Object2"], ...], // Changed from new_identifiers
            "contextual_value": "Brief explanation of how this information contributes to understanding {self.topic} or its context"
        }}
        """
        
        try:
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
                result = json.loads(cleaned_response)
                
                # Check if content should be accepted based on relevance criteria
                should_accept = result.get("should_accept", True)  # Default to True to be more accepting
                
                # If the content should be accepted, process it
                if should_accept:
                    # Add new triplets to knowledge graph
                    new_triplets = result.get("new_triplets", []) # Changed from new_identifiers
                    if isinstance(new_triplets, list):
                        # Use tuples for checking existence in the set for hashability
                        existing_triplets = {tuple(triplet) for triplet in self.knowledge_graph["triplets"]}
                        
                        added_count = 0
                        for triplet in new_triplets:
                            if isinstance(triplet, list) and len(triplet) == 3:
                                # Check if the tuple version of the triplet exists
                                if tuple(triplet) not in existing_triplets:
                                    self.knowledge_graph["triplets"].append(triplet)
                                    existing_triplets.add(tuple(triplet)) # Add to set to prevent duplicates within the same chunk processing
                                    added_count += 1
                            else:
                                logging.warning(f"Skipping invalid triplet format: {triplet}")
                        if added_count > 0:
                             logging.info(f"Added {added_count} new triplets.")
                
                # Optimize knowledge graph when we have too many triplets
                # NOTE: optimize_knowledge_graph needs to be updated to handle triplets
                if len(self.knowledge_graph["triplets"]) > 20: 
                    self.knowledge_graph = optimize_knowledge_graph(self.knowledge_graph, self.llm, topic=self.topic)
                
                return result
            except json.JSONDecodeError:
                logging.error(f"Failed to parse response as JSON: {response}")
                return None
        except Exception as e:
            logging.error(f"Error processing chunk from {url}: {str(e)}")
            logging.exception(e)
            return None
    
    def get_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the current knowledge graph triplets and URL classification.
        
        Returns:
            Dictionary with 'triplets', 'accepted_urls', and 'rejected_urls' lists
        """
        return {
            "triplets": self.knowledge_graph["triplets"], # Return triplets
            "accepted_urls": list(self.accepted_urls),
            "rejected_urls": list(self.rejected_urls),
            "rejection_reasons": self.url_rejection_reasons
        }
        
    def save_url_classification(self, output_dir: str) -> None:
        """
        Save the accepted and rejected URLs to a file in the specified output directory.
        
        Args:
            output_dir: Directory to save the URL classification file
        """
        import os
        import json
        
        url_classification = {
            "topic": self.topic,
            "accepted_urls": list(self.accepted_urls),
            "rejected_urls": list(self.rejected_urls),
            "rejection_reasons": self.url_rejection_reasons
        }
        
        output_path = os.path.join(output_dir, "url_classification.json")
        with open(output_path, "w") as f:
            json.dump(url_classification, f, indent=2)
            
        logging.info(f"Saved URL classification to {output_path}")
        logging.info(f"Accepted URLs: {len(self.accepted_urls)}, Rejected URLs: {len(self.rejected_urls)}")

# NOTE: This function needs to be updated to handle triplets instead of string facts
def optimize_knowledge_graph(knowledge_graph: Dict[str, List[Any]], llm: Callable, topic: Optional[str] = None) -> Dict[str, List[Any]]:
    """
    Optimize the knowledge graph by condensing and prioritizing the most unique and informative triplets.
    (Needs update to handle triplets)
    
    Args:
        knowledge_graph: The current knowledge graph (expects 'triplets' key)
        llm: A callable that takes a prompt and returns a response
        topic: Optional topic for context
        
    Returns:
        Optimized knowledge graph
    """
    triplets = knowledge_graph.get("triplets", []) # Changed from facts
    
    # Always optimize if we have more than 15 triplets to keep the knowledge graph concise
    if len(triplets) <= 15: # Changed from facts
        return knowledge_graph
    
    # NOTE: The logic below assumes facts are strings. It needs a complete rewrite for triplets.
    # The prompt needs to ask the LLM to consolidate/summarize/prioritize triplets.
    
    # Placeholder: Return original graph until updated
    logging.warning("optimize_knowledge_graph needs to be updated to handle triplets. Returning original graph.")
    return knowledge_graph

    # # Standardize facts to strings if they are dictionaries - REMOVE/REPLACE THIS LOGIC
    # standardized_facts = []
    # for fact in facts:
    #     if isinstance(fact, dict) and "fact" in fact:
    #         standardized_facts.append(fact["fact"])
    #     elif isinstance(fact, str):
    #         standardized_facts.append(fact)
    
    # prompt = f"""
    # I have collected the following facts about "{topic if topic else 'this entity'}", 
    # but there are too many of them. Please identify the 10-15 MOST unique, specific, and 
    # distinguishing facts that would help identify this entity and differentiate it from similar entities.
    
    # Prioritize facts that contain:
    # 1. Exact dates/timeframes linked to specific, unique events *central* to this entity's identity or history.
    # 2. Specific names of key individuals/leaders and their *defining* roles/actions within this entity.
    # 3. Exact locations/territory uniquely associated with this entity's core operations or defining events.
    # 4. Precise numerical data uniquely relevant to *this entity's* scale, actions, or impact.
    # 5. Specific events or incidents *central* to this entity's identity or history.
    # 6. Specific distinguishing activities, tactics, or methods *unique* to this entity compared to peers.
    # 7. Defining relationships (affiliations, parent bodies, key rivals, governance structure) that are critical to understanding this entity's unique position.
    # 8. The core mandate, purpose, or defining activity that sets this entity apart.
    
    # BE EXTREMELY SELECTIVE - only keep the MOST specific and unique facts.
    # DO NOT include facts that are general for entities of this type or region.
    # DO NOT include broad categorical information unless highly specified.
    # PRIORITIZE facts with concrete details, specific references, and exact information.
    
    # Current facts:
    # {json.dumps(standardized_facts)}
    
    # Return ONLY a JSON array containing the 10-15 most specific and unique facts:
    # """ # END OF OLD PROMPT - NEEDS REPLACEMENT FOR TRIPLETS
    
    # try:
    #     # Get response from LLM
    #     response = llm(prompt) # OLD PROMPT
        
    #     # Handle list response (some LLMs return a list with one element)
    #     if isinstance(response, list) and len(response) > 0:
    #         response = response[0]
        
    #     # Process the response
    #     try:
    #         # Extract JSON content from response
    #         cleaned_response = extract_json_from_response(response)
            
    #         # Parse the JSON response
    #         optimized_facts = json.loads(cleaned_response) # OLD LOGIC
            
    #         # Update knowledge graph with optimized facts
    #         if isinstance(optimized_facts, list) and len(optimized_facts) > 0:
    #             result = {"facts": optimized_facts} # OLD LOGIC
    #             logging.info(f"Optimized knowledge graph from {len(facts)} to {len(optimized_facts)} facts") # OLD LOGIC
    #             return result
    #         else:
    #             logging.error(f"Unexpected optimize_knowledge_graph result format: {optimized_facts}") # OLD LOGIC
    #             return knowledge_graph
    #     except json.JSONDecodeError:
    #         logging.error(f"Failed to parse optimize_knowledge_graph response as JSON: {response}") # OLD LOGIC
    #         return knowledge_graph
    # except Exception as e:
    #     logging.error(f"Error optimizing knowledge graph: {str(e)}") # OLD LOGIC
    #     logging.exception(e)
    #     return knowledge_graph # OLD LOGIC
