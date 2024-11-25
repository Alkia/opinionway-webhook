def summarize_opinion_with_llm(opinion: str) -> str:
    """
    Summarizes the opinion into a clear, concise single sentence.
    
    :param opinion: The full opinion to be summarized.
    :return: A clear, single-idea summary of the opinion.
    """
    if not opinion.strip():
        return ""
    
    if not summarizer:
        return extract_main_clause(opinion)
    
    try:
        # Parse the text
        parser = PlaintextParser.from_string(opinion, Tokenizer('english'))
        
        # Generate summary (get one sentence)
        summary = summarizer(parser.document, sentences_count=1)
        
        if not summary:
            return extract_main_clause(opinion)
            
        # Extract the main clause from the summary
        summary_text = str(summary[0])
        return extract_main_clause(summary_text)
        
    except Exception as e:
        logging.error(f"Summarization failed: {str(e)}")
        return extract_main_clause(opinion)

def extract_main_clause(text: str) -> str:
    """
    Extracts the main clause from a sentence to create a clear, focused summary.
    
    :param text: The input text to process.
    :return: A single, clear statement.
    """
    # Process the text with spaCy
    doc = nlp(text)
    
    # Find the main verb and its associated subject and object
    main_parts = []
    for token in doc:
        if token.dep_ == "ROOT":  # Main verb
            # Get subject
            subject = next((t for t in token.lefts if t.dep_ == "nsubj"), None)
            if subject:
                main_parts.extend([t.text for t in subject.subtree])
            
            # Get the verb
            main_parts.append(token.text)
            
            # Get direct object or complement
            for right_token in token.rights:
                if right_token.dep_ in ["dobj", "attr", "ccomp"]:
                    main_parts.extend([t.text for t in right_token.subtree])
                    break
            
            break
    
    if not main_parts:
        # Fallback: take the first 8 words
        return " ".join(text.split()[:8])
    
    # Join the main parts and ensure it's not too long
    summary = " ".join(main_parts)
    return " ".join(summary.split()[:12])

def classify_opinion(opinion: str) -> str:
    """
    Classify the sentiment of an opinion with improved accuracy.
    
    :param opinion: Opinion text.
    :return: Sentiment classification as a string.
    """
    opinion = opinion.lower()
    
    # Enhanced keyword lists with weights
    sentiment_markers = {
        "positive": {
            "smooth": 1,
            "accessible": 1,
            "appealing": 1,
            "convenient": 1,
            "positive": 1,
            "easy": 1,
            "stress-free": 1,
            "innovative": 1
        },
        "negative": {
            "misleading": 2,  # Higher weight for strong negative indicators
            "risk": 1.5,
            "liquidation": 1.5,
            "complex": 1,
            "risky": 1.5,
            "stressful": 1,
            "bad": 1,
            "dangerous": 1.5,
            "difficult": 1
        }
    }
    
    # Calculate sentiment scores
    positive_score = sum(weight for word, weight in sentiment_markers["positive"].items() 
                        if word in opinion)
    negative_score = sum(weight for word, weight in sentiment_markers["negative"].items() 
                        if word in opinion)
    
    # Check for negation words that could flip the sentiment
    negation_words = ["not", "no", "never", "neither", "nor", "without"]
    contains_negation = any(word in opinion.split() for word in negation_words)
    
    if contains_negation:
        # Flip the scores if negation is present
        positive_score, negative_score = negative_score, positive_score
    
    # Determine classification with a neutral threshold
    if positive_score > negative_score and positive_score > 0:
        return "Positive"
    elif negative_score > positive_score and negative_score > 0:
        return "Negative"
    else:
        return "Neutral"

def extract_themes_and_classify(data: dict) -> List[Dict[str, List[Dict[str, str]]]]:
    """
    Extract themes, opinions, and classify them with improved accuracy.
    
    :param data: Dictionary containing the input transcript data.
    :return: List of themes with their respective opinions and classifications.
    """
    themes_of_interest = [
        {
            "theme": "Centralized exchanges",
            "keywords": ["centralized exchange", "smooth user experience", "trading fee", "bybit", "exchange"],
            "context_words": ["platform", "trading", "exchange"]
        },
        {
            "theme": "Lending products",
            "keywords": ["lending", "loan", "stake", "unstake", "savings"],
            "context_words": ["interest", "yield", "term"]
        },
        {
            "theme": "Dual asset strategies",
            "keywords": ["dual asset", "options trading", "derivatives"],
            "context_words": ["APR", "premium", "trading"]
        }
    ]

    result = []
    
    transcript_segments = data.get("payload", {}).get("transcript_segments", [])
    
    for segment in transcript_segments:
        text = segment.get("text", "")
        normalized_text = normalize_text(text)
        
        for theme in themes_of_interest:
            # Check both keywords and context words
            keyword_match = any(keyword.lower() in normalized_text for keyword in theme['keywords'])
            context_match = any(word.lower() in normalized_text for word in theme['context_words'])
            
            if keyword_match and context_match:
                opinion = text.strip()
                summarized_theme = summarize_theme(theme["theme"])
                summarized_opinion = summarize_opinion_with_llm(opinion)
                classification = classify_opinion(opinion)
                
                # Only add if we have a meaningful opinion
                if len(summarized_opinion.split()) >= 3:
                    log_json_data(summarized_theme, summarized_opinion, classification)
                    result.append({
                        "theme": summarized_theme,
                        "opinion": summarized_opinion,
                        "classification": classification
                    })

    return result