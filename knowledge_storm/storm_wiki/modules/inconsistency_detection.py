import logging
import json
from typing import List, Dict, Any, Optional, Union

import dspy

from ...interface import Information

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InconsistencyDetection")

class SearchVerifier:
    """
    Verifies information about a specific entity from search results.
    
    This class maintains a knowledge graph of verified facts and inconsistencies.
    """
    
    def __init__(self, topic: str, llm: Any):
        """
        Initialize the SearchVerifier with a topic and language model.
        
        Args:
            topic: The main entity or topic to track
            llm: Language model instance for verification
        """
        self.topic = topic
        self.llm = llm
        # Initialize knowledge graph with empty facts and inconsistencies
        self.knowledge_graph = {
            "facts": [],  # List of verified facts with their sources
            "inconsistencies": []  # List of inconsistencies with their sources and reasoning
        }
        logger.info(f"Initialized SearchVerifier for topic: {topic}")
        
    def verify_and_enrich(self, content: dict) -> dict:
        """
        Verify new content against existing knowledge and enrich the knowledge graph.
        
        Args:
            content: Dict containing 'url', 'title', 'description', and 'snippets'
            
        Returns:
            dict: The original content with verification results
        """
        url = content.get('url', 'Unknown URL')
        logger.info(f"Verifying content from URL: {url}")
        
        # Process each snippet
        for snippet in content['snippets']:
            # Verify the snippet
            self._verify_chunk(snippet, url)
        
        # Add verification metadata to content
        content['verification'] = {
            "verified": True
        }
        
        return content
        
    def _verify_chunk(self, chunk: str, url: str) -> None:
        """
        Verify a chunk of text against the knowledge graph.
        
        Args:
            chunk: The text chunk to verify
            url: Source URL of the chunk
        """
        # Prepare context for verification
        context = self._format_knowledge()
        
        # Create prompt for verification
        prompt = f"""You are verifying information about {self.topic}.

Current knowledge:
{context}

New information from {url}:
{chunk}

Analyze this information and determine:
1. If it contains factual information about {self.topic} (confirm it's about the same entity)
2. If it contains any inconsistencies or contradictions with existing knowledge
3. What new facts can be extracted
4. Pay special attention to dates, numbers, and specific claims that might conflict

Respond in JSON format:
{{
    "new_facts": [],
    "inconsistencies": []
}}

For inconsistencies, include the exact inconsistency and the reason it's inconsistent.
DO NOT include inconsistencies and facts that are already in the current knowledge.
"""
        
        try:
            # Get verification from LLM
            response = self.llm(prompt)
            
            # The response is a list, so we need to get the first completion
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0]
            else:
                response_text = str(response)
            
            # Check if the response is empty or not valid JSON
            if not response_text or response_text.strip() == "":
                logger.warning("Empty response from language model")
                return
            
            # Try to parse as JSON
            try:
                json_result = json.loads(response_text)
                
                # Process inconsistencies
                if json_result.get("inconsistencies"):
                    for inconsistency in json_result["inconsistencies"]:
                        # Check if this inconsistency is already in our knowledge graph
                        is_duplicate = False
                        for existing in self.knowledge_graph["inconsistencies"]:
                            if existing["inconsistency"] == inconsistency:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            self.knowledge_graph["inconsistencies"].append({
                                "inconsistency": inconsistency,
                                "url": url
                            })
                
                # Process new facts
                if json_result.get("new_facts"):
                    for fact in json_result["new_facts"]:
                        # Check if this fact is already in our knowledge graph
                        is_duplicate = False
                        for existing in self.knowledge_graph["facts"]:
                            if existing["fact"] == fact:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            self.knowledge_graph["facts"].append({
                                "fact": fact,
                                "url": url
                            })
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}. Response text: {response_text}")
            
        except Exception as e:
            logger.error(f"Error verifying chunk: {str(e)}")
        print('self.knowledge_graph: ', self.knowledge_graph)
            
    def _format_knowledge(self) -> str:
        """
        Format the current knowledge for LLM context.
        
        Returns:
            str: Formatted knowledge as a string
        """
        facts = "\n".join([f"- {item['fact']}" for item in self.knowledge_graph["facts"]])
        inconsistencies = "\n".join([f"- {item['inconsistency']}" for item in self.knowledge_graph["inconsistencies"]])
        
        return f"""Verified Facts:
{facts}

Known Inconsistencies:
{inconsistencies}"""
    
    def get_flagged_claims(self) -> List[str]:
        """
        Get all flagged claims from the knowledge graph.
        
        Returns:
            List of flagged claims
        """
        return [item["inconsistency"] for item in self.knowledge_graph["inconsistencies"]]


class InconsistencyDetector:
    """
    A class for detecting and resolving inconsistencies in information retrieved from multiple sources.
    """
    
    def __init__(self, llm: dspy.dsp.LM):
        """
        Initialize the inconsistency detector with a language model.
        
        Args:
            llm: The language model to use for inconsistency detection and resolution.
        """
        self.llm = llm
        # Initialize DSPy modules properly
        # self.detect_inconsistencies = dspy.Predict(DetectInconsistencies)
        self.resolve_inconsistencies = dspy.Predict(ResolveInconsistencies)
    
    def process_information(self, information_list: List[Information], flagged_claims: List[str] = None) -> List[Information]:
        """
        Process information to handle inconsistencies.
        
        Args:
            information_list: List of Information objects to process.
            flagged_claims: Optional list of claims flagged as potentially inconsistent.
            
        Returns:
            List of processed Information objects.
        """
        logger.info(f"Processing {len(information_list)} information objects for inconsistencies")
        
        # If no flagged claims are provided, detect potential inconsistencies
        if not flagged_claims:
            # Extract all claims from the information
            # all_claims = []
            # for info in information_list:
            #     for snippet in info.snippets:
            #         all_claims.append(snippet)
            
            # # Format claims as a string for the DSPy module
            # formatted_claims = "\n".join([f"Claim {i+1}: \"{claim}\"" for i, claim in enumerate(all_claims)])
            
            # # Detect inconsistencies using DSPy
            # with dspy.settings.context(lm=self.llm):
            #     inconsistency_results = self.detect_inconsistencies(
            #         claims=formatted_claims
            #     )
            
            # # Extract flagged claims
            # flagged_claims = inconsistency_results.flagged_claims
            logger.info(f"Detected 0 potentially inconsistent claims")
        else:
            print(flagged_claims)
            logger.info(f"Using {len(flagged_claims)} flagged claims from SearchVerifier")
        
        # If there are flagged claims, resolve them
        if flagged_claims:
            # Group information by URL for easier processing
            url_to_info = {}
            for info in information_list:
                url_to_info[info.url] = info
            
            # Resolve inconsistencies for each flagged claim
            for claim in flagged_claims:
                # Find all information objects that contain the claim
                claim_info = []
                for info in information_list:
                    for snippet in info.snippets:
                        if claim in snippet:
                            claim_info.append((info, snippet))
                
                if len(claim_info) > 1:
                    logger.info(f"Resolving inconsistency for claim: {claim}")
                    
                    # Prepare information for resolution
                    resolution_info = []
                    for info, snippet in claim_info:
                        resolution_info.append({
                            "url": info.url,
                            "title": info.title,
                            "snippet": snippet,
                            "source_trust": self._estimate_source_trust(info)
                        })
                    
                    # Resolve the inconsistency using DSPy
                    with dspy.settings.context(lm=self.llm):
                        resolution = self.resolve_inconsistencies(
                            claim=claim,
                            information=resolution_info
                        )
                    
                    # Update the information based on the resolution
                    if resolution.resolved_claim:
                        # If we have a clear resolution, remove the inconsistent claims
                        # and keep only the resolved claim
                        for info, snippet in claim_info:
                            if snippet != resolution.resolved_claim:
                                info.snippets.remove(snippet)
                        
                        # Add the resolved claim to the most trusted source
                        if resolution.resolved_claim not in url_to_info[resolution.trusted_source].snippets:
                            url_to_info[resolution.trusted_source].snippets.append(resolution.resolved_claim)
                    else:
                        # If we can't determine which is correct, flag all claims as potentially conflicting
                        for info, snippet in claim_info:
                            # Remove the original snippet
                            info.snippets.remove(snippet)
                            # Add the flagged snippet
                            flagged_snippet = f"POTENTIALLY CONFLICTING: {snippet}"
                            info.snippets.append(flagged_snippet)
                            logger.info(f"Flagged potentially conflicting claim: {snippet}")
        
        return information_list
    
    def _estimate_source_trust(self, info: Information) -> float:
        """
        Estimate the trustworthiness of a source based on various factors.
        
        Args:
            info: The information object containing the source.
            
        Returns:
            A trust score between 0 and 1.
        """
        # This is a simple implementation that could be enhanced
        # For now, we'll use a basic heuristic based on URL domain
        
        url = info.url.lower()
        
        # Higher trust for academic and government domains
        if any(domain in url for domain in [".edu", ".gov", ".org"]):
            return 0.9
        
        # Medium trust for news sites
        if any(domain in url for domain in [".com", ".net"]):
            return 0.7
        
        # Lower trust for others
        return 0.5


class DetectInconsistencies(dspy.Signature):
    """
    Detect inconsistencies in claims.
    """
    claims = dspy.InputField(desc="List of claims to check for inconsistencies")
    flagged_claims = dspy.OutputField(desc="List of claims that are potentially inconsistent")
    
    def _format_claims(self, claims: List[str]) -> str:
        """
        Format claims for the prompt.
        
        Args:
            claims: List of claims to format.
            
        Returns:
            Formatted claims string.
        """
        formatted_claims = ""
        for i, claim in enumerate(claims):
            formatted_claims += f"Claim {i+1}: \"{claim}\"\n"
        return formatted_claims


class ResolveInconsistencies(dspy.Signature):
    """
    Resolve inconsistencies between multiple sources of information.
    """
    claim = dspy.InputField(desc="The claim that needs to be resolved")
    information = dspy.InputField(desc="List of information objects containing the claim")
    resolved_claim = dspy.OutputField(desc="The resolved version of the claim, or empty string if unable to resolve")
    trusted_source = dspy.OutputField(desc="URL of the most trusted source, or empty string if unable to determine")
    explanation = dspy.OutputField(desc="Explanation of how the inconsistency was resolved, or 'Unable to determine which claim is correct' if unable to resolve")
    
    def _format_sources(self, information: List[Dict[str, Any]]) -> str:
        """
        Format source information for the prompt.
        
        Args:
            information: List of dictionaries containing source information
            
        Returns:
            Formatted string of source information
        """
        formatted_sources = []
        for i, source in enumerate(information, 1):
            formatted_sources.append(f"""Source {i}:
URL: {source['url']}
Title: {source['title']}
Content: {source['snippet']}
Trust Score: {source['source_trust']}
""")
        return "\n".join(formatted_sources) 