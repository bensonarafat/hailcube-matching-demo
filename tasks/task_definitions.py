from services.celery_service import celery
import pandas as pd

# Import your services after celery is defined
from services.matching_system import MatchingSystem
from services.messaging_system import MessagingSystem

@celery.task(bind=True, max_retries=3)
def send_message(self, contact_dict, opportunity_dict):
    try: 
        ms = MessagingSystem()
        ms.send_message(pd.Series(contact_dict), pd.Series(opportunity_dict)) 
    except Exception as e: 
        self.retry(exc=e, countdown=2 ** self.request.retries)
        
@celery.task
def run_matching():
    min_similarity_threshold = 0.15
    top_n = 3
    """
    Run the matching process as a background task with proper error handling
    """
    
    print("Start Running the background task")
    
    try:
        # Initialize matching system with threshold
        matcher = MatchingSystem(min_similarity_threshold=min_similarity_threshold)
        
        # Load data with validation
        if not matcher.load_data():
            print("Error loading data. Aborting matching process.")
            return False 
        
        new_opps = matcher.opportunities
        
        if new_opps.empty:
            print("No opportunities found, Skipping matching process.") 
            return True 
        
        #   Generate embadding
        if not matcher.generate_embeddings():
            print("Error generating embeddings. Aborting matching process.") 
            return False 
        
        # Find matches using the specified top_n
        matches = matcher.find_matches(top_n=top_n)
        match_count = 0
        
        # Process each match and send messages 
        for opp_id, contacts_df in matches.items():
            if contacts_df.empty:
                print(f"No suitable matches found for opportunity {opp_id}") 
                continue 
            
            try:
                opportunity = matcher.opportunities[matcher.opportunities['id'] == opp_id].iloc[0]
                opportunity_dict = opportunity.to_dict()
                
                for _, contact in contacts_df.iterrows(): 
                    contact_dict = contact.to_dict()
                    #  Include match similarity score in the message data
                    contact_dict['match_score'] = float(contact['similarity_score'])
                    # Send message asynchronously
                    send_message.delay(contact_dict, opportunity_dict)
                    match_count += 1
            except IndexError:
                print(f"Could not find opportunity with ID {opp_id}") 
            except Exception as e: 
                print(f"Error processing match for opportunity {opp_id}: {str(e)}")
        print(f"Matching process completed. Sent {match_count} match notifications.")
        return True
    except Exception as e:
        print(f"Unexpected error in matching process: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False