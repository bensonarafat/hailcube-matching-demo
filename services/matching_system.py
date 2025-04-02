import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re

class MatchingSystem:
    def __init__(self, min_similarity_threshold=0.1):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
        self.min_similarity_threshold = min_similarity_threshold
        
    def load_data(self, contacts_file='data/contacts.csv', opportunities_file='data/opportunities.csv'):
         """Load data from CSV files with more robust error handling"""
         try:
           self.contacts = pd.read_csv(contacts_file)
           self.opportunities = pd.read_csv(opportunities_file, dtype={'id': str})
           
           print(f"Loaded {len(self.contacts)} contacts and {len(self.opportunities)} opportunities")
            
           # Validate required columns
           required_contact_cols = ['skills', 'timezone']
           required_opp_cols = ['id', 'description', 'required_skills', 'urgency']
           
           for col in required_contact_cols:
               if col not in self.contacts.columns:
                   raise ValueError(f"Missing required column '{col} in contacts data")
           
           for col in required_opp_cols:
               if col not in self.opportunities.columns:
                   raise ValueError(f"Missing required column '{col} in opportunities data")
               
           # Filter to only include active contacts (status=1)
           self.contacts = self.contacts[self.contacts['status'] == 1]
           print(f"Using {len(self.contacts)} active contacts for matching")

           return True 
         except Exception as e:
            print(f"Error loading data: {type(e).__name__}: {str(e)}")
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize, remove stopwords, and stem
        words = [self.stemmer.stem(word) for word in text.split() if word not in self.stop_words]
        
        return " ".join(words)
    
    def generate_embeddings(self):
        """Generate embedding with weighted features and preprocessing"""
        try:
            print("Starting embedding generation...") 
          
            #   Handle NaN values 
            for col in ['skills', 'timezone']:
                if col in self.contacts.columns:
                    self.contacts[col] = self.contacts[col].fillna('')
            
            for col in ['description', 'required_skills', 'urgency']:
                if col in self.opportunities.columns:
                    self.opportunities[col] = self.opportunities[col].fillna('')
            
            # Preprocess text fields 
            self.contacts['skills_processed'] = self.contacts['skills'].apply(self.preprocess_text)
            self.contacts['timezone_processed'] = self.contacts['timezone'].apply(self.preprocess_text)
            
            self.opportunities['description_processed'] = self.opportunities['description'].apply(self.preprocess_text)
            self.opportunities['skills_processed'] = self.opportunities['required_skills'].apply(self.preprocess_text)
            self.opportunities['urgency_processed'] = self.opportunities['urgency'].apply(self.preprocess_text)

            # Combine text features with weighting (skills are most important)
            self.contacts['text'] = (
                self.contacts['skills_processed'] + " " + 
                self.contacts['skills_processed'] + " " + # Repeat for higher weight
                self.contacts['timezone_processed']
            )
            
            self.opportunities['text'] = (
                self.opportunities['skills_processed'] + " " +
                self.opportunities['skills_processed'] + " " +  # Repeat for higher weight
                self.opportunities['description_processed'] + " " +
                self.opportunities['urgency_processed']
            )
            
            # Combine all texts to fit the vectorizer
            all_texts = self.contacts['text'].tolist() + self.opportunities['text'].tolist()
            self.vectorizer.fit(all_texts)
            
            # Generate embeddings
            self.contact_embeddings = self.vectorizer.transform(self.contacts['text'].tolist()).toarray()
            self.opportunity_embeddings = self.vectorizer.transform(self.opportunities['text'].tolist()).toarray()

            # Store feature names for later analysis
            self.feature_names = self.vectorizer.get_feature_names_out()
                
            print(f"Embedding generation complete: {len(self.contact_embeddings)} contacts, {len(self.opportunity_embeddings)} opportunities")
            return True
    
        except Exception as e:
            print(f"Error in generate_embeddings: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_matches(self, top_n=3):
        """Find matches with similarity scores and threshold"""
        similarity_matrix = cosine_similarity(self.opportunity_embeddings, self.contact_embeddings)
        matches = {}
        
        for opp_idx, opp_row in self.opportunities.iterrows():
            opp_id = opp_row['id']
            
            # Get similarity scores 
            similarity_scores = similarity_matrix[opp_idx]
            
            # Create a list of (index, score) tuples 
            scored_indices = [(i, score) for i, score in enumerate(similarity_scores)]
            
            # Sort by score in desending order 
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by minimum threshold and take top_n
            filtered_indices = [idx for idx, score in scored_indices if score >= self.min_similarity_threshold][:top_n]
            
            # Get the corresponding contacts and scores 
            match_contacts = self.contacts.iloc[filtered_indices].copy()
            
            # Add similarity scores to the results 
            if not match_contacts.empty:
                match_scores = [similarity_scores[idx] for idx in filtered_indices]
                match_contacts['similarity_score'] = match_scores
                matches[opp_id] = match_contacts
            else:
                matches[opp_id] = pd.DataFrame()
                
        return matches