from flask import Flask, render_template, request, jsonify, send_from_directory, session, make_response, redirect, url_for
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
import shutil
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import json
import requests
from datetime import datetime
import uuid
import re
from functools import wraps
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import csv
from io import StringIO

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, 
    static_url_path='', 
    static_folder='static',
    template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Initialize global variables
llm = None
user_preferences = {}  # Store user preferences

# Enhanced conversation state management
class ConversationContext:
    def __init__(self):
        self.user_profile = {}
        self.farming_context = {}
        self.conversation_stage = "greeting"
        self.last_topic = None
        self.follow_up_suggestions = []
        self.user_expertise_level = "beginner"
        
    def update_context(self, user_input, bot_response):
        """Update conversation context based on interaction"""
        # Extract farming-related keywords
        farming_keywords = {
            'crops': ['maize', 'tobacco', 'cotton', 'wheat', 'sorghum', 'millet', 'rice', 'beans'],
            'activities': ['planting', 'harvesting', 'irrigation', 'fertilizer', 'pest', 'disease'],
            'seasons': ['summer', 'winter', 'autumn', 'spring', 'rainy', 'dry'],
            'locations': ['zimbabwe', 'zambia', 'botswana', 'namibia', 'south africa', 'mozambique']
        }
        
        user_lower = user_input.lower()
        
        # Update farming context
        for category, keywords in farming_keywords.items():
            for keyword in keywords:
                if keyword in user_lower:
                    if category not in self.farming_context:
                        self.farming_context[category] = []
                    if keyword not in self.farming_context[category]:
                        self.farming_context[category].append(keyword)
        
        # Determine conversation stage
        if any(word in user_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            self.conversation_stage = "greeting"
        elif any(word in user_lower for word in ['plant', 'grow', 'crop', 'seed']):
            self.conversation_stage = "crop_planning"
        elif any(word in user_lower for word in ['pest', 'disease', 'problem', 'help']):
            self.conversation_stage = "problem_solving"
        elif any(word in user_lower for word in ['weather', 'rain', 'temperature', 'climate']):
            self.conversation_stage = "weather_inquiry"
        elif any(word in user_lower for word in ['harvest', 'storage', 'sell', 'market']):
            self.conversation_stage = "harvest_marketing"
        
        # Assess expertise level
        technical_terms = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'hectare', 'yield']
        if any(term in user_lower for term in technical_terms):
            self.user_expertise_level = "intermediate"
        elif any(word in user_lower for word in ['fertilizer', 'pesticide', 'irrigation']):
            self.user_expertise_level = "beginner_plus"

# Define seasons in Southern Africa
def get_current_season():
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return "autumn"
    elif 6 <= current_month <= 8:
        return "winter"
    elif 9 <= current_month <= 11:
        return "spring"
    else:
        return "summer"

# Define common crops by season in Southern Africa
seasonal_crops = {
    "summer": ["maize", "sorghum", "millet", "groundnuts", "cotton", "soybeans", "sunflower", "tobacco", "vegetables"],
    "autumn": ["winter wheat", "barley", "potatoes", "vegetable harvest", "land preparation"],
    "winter": ["wheat", "barley", "oats", "peas", "leafy greens", "onions", "garlic"],
    "spring": ["maize preparation", "tobacco seedbeds", "cotton preparation", "vegetable planting", "soil preparation"]
}

# MongoDB Connection Setup
def setup_mongodb():
    """Setup MongoDB connection"""
    try:
        # Get MongoDB URI from environment variables with fallback
        mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
        
        # Connect to MongoDB with connection pooling
        client = MongoClient(
            mongodb_uri,
            server_api=ServerApi('1'),
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=30000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
            serverSelectionTimeoutMS=5000
        )
        
        # Test connection
        client.admin.command('ping')
        print("MongoDB connection successful")
        
        # Set up database and collections
        db = client['mudhumeni_db']
        crop_recommendations = db['crop_recommendations']
        
        # Create indexes for better performance
        crop_recommendations.create_index([("user_id", 1)])
        crop_recommendations.create_index([("recommendation_date", -1)])
        crop_recommendations.create_index([
            ("outputs.predicted_crop", 1),
            ("inputs.province", 1)
        ])
        
        return {
            "client": client,
            "db": db,
            "collections": {
                "crop_recommendations": crop_recommendations
            }
        }
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        return None

# Enhanced conversation memory management
class EnhancedConversationMemory:
    def __init__(self, max_turns=20):
        self.max_turns = max_turns
        self.conversation_history = []
        self.context = ConversationContext()
        self.key_insights = []
        
    def add_exchange(self, user_input, bot_response):
        """Add a conversation exchange and update context"""
        exchange = {
            "timestamp": datetime.now(),
            "user": user_input,
            "bot": bot_response,
            "stage": self.context.conversation_stage
        }
        
        self.conversation_history.append(exchange)
        self.context.update_context(user_input, bot_response)
        
        # Keep only recent history to manage memory
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]
            
        # Extract key insights
        self._extract_insights(user_input, bot_response)
    
    def _extract_insights(self, user_input, bot_response):
        """Extract key insights from conversation"""
        user_lower = user_input.lower()
        
        # Extract farming preferences and context
        if 'farm' in user_lower and ('hectare' in user_lower or 'acre' in user_lower):
            size_match = re.search(r'(\d+(?:\.\d+)?)\s*(hectare|acre)', user_lower)
            if size_match:
                self.key_insights.append(f"Farm size: {size_match.group(1)} {size_match.group(2)}")
        
        if any(crop in user_lower for crop in ['maize', 'tobacco', 'cotton', 'wheat']):
            crops = [crop for crop in ['maize', 'tobacco', 'cotton', 'wheat'] if crop in user_lower]
            self.key_insights.append(f"Interested in crops: {', '.join(crops)}")
    
    def get_context_summary(self):
        """Get a summary of the conversation context"""
        if not self.conversation_history:
            return ""
            
        summary = []
        
        # Add farming context
        if self.context.farming_context:
            if 'crops' in self.context.farming_context:
                summary.append(f"Crops discussed: {', '.join(self.context.farming_context['crops'])}")
            if 'activities' in self.context.farming_context:
                summary.append(f"Activities: {', '.join(self.context.farming_context['activities'])}")
        
        # Add key insights
        if self.key_insights:
            summary.extend(self.key_insights[-3:])  # Last 3 insights
            
        return " | ".join(summary)
    
    def get_recent_context(self, num_exchanges=5):
        """Get recent conversation context"""
        recent = self.conversation_history[-num_exchanges:] if self.conversation_history else []
        context_text = ""
        
        for exchange in recent:
            context_text += f"User: {exchange['user']}\nAssistant: {exchange['bot']}\n"
            
        return context_text

# Add a route for multilingual support - translate response to a supported language
@app.route('/translate', methods=['POST'])
def translate_response():
    try:
        data = request.json
        text = data.get('text', '')
        target_language = data.get('language', 'en')  # Default to English
        
        # Supported languages in Southern Africa
        supported_languages = {
            'en': 'English',
            'af': 'Afrikaans',
            'st': 'Sesotho',
            'tn': 'Setswana',
            'xh': 'isiXhosa',
            'zu': 'isiZulu',
            'sn': 'Shona',
            'nd': 'Ndebele',
            'sw': 'Swahili',
            'pt': 'Portuguese'  # For Mozambique, Angola
        }
        
        # For now, this is a mock translation as integrating a real translation API would require additional setup
        # You could integrate Google Translate API or similar service here
        
        if target_language != 'en' and target_language in supported_languages:
            # This is a mock - in production, you would call a translation API
            translated_text = f"[This would be translated to {supported_languages[target_language]}]: {text}"
            return jsonify({'translated_text': translated_text, 'language': target_language})
        else:
            return jsonify({'translated_text': text, 'language': 'en'})
    except Exception as e:
        return jsonify({'error': str(e)})

# Add a route to set user preferences
@app.route('/set_preferences', methods=['POST'])
def set_user_preferences():
    try:
        data = request.json
        user_id = session.get('user_id')
        
        if not user_id:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
        
        if user_id not in user_preferences:
            user_preferences[user_id] = {}
        
        # Update preferences
        if 'location' in data:
            user_preferences[user_id]['location'] = sanitize_input(data['location'])
        
        if 'farming_type' in data:
            user_preferences[user_id]['farming_type'] = sanitize_input(data['farming_type'])
        
        return jsonify({
            'success': True, 
            'message': 'Preferences updated successfully',
            'preferences': user_preferences[user_id]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Clear chat history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['chat_history'] = []
    # Also clear enhanced memory
    user_id = session.get('user_id')
    if user_id and f"{user_id}_memory" in session:
        del session[f"{user_id}_memory"]
    return jsonify({'success': True, 'message': 'Chat history cleared'})

# Analytics Dashboard routes (keeping existing analytics functionality)
@app.route('/admin/analytics')
def analytics_dashboard():
    """Admin analytics dashboard"""
    return render_template('analytics.html')

@app.route('/api/analytics/summary', methods=['GET'])
def analytics_summary():
    """API endpoint for summary statistics"""
    try:
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        # Get total recommendations
        total_recommendations = mongo_data['collections']['crop_recommendations'].count_documents({})
        
        # Get unique users
        pipeline = [{"$group": {"_id": "$user_id"}}, {"$count": "total"}]
        unique_users_result = list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
        unique_users = unique_users_result[0]['total'] if unique_users_result else 0
        
        # Get top crop
        pipeline = [
            {"$group": {"_id": "$outputs.predicted_crop", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 1}
        ]
        top_crop_result = list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
        top_crop = top_crop_result[0]['_id'] if top_crop_result else "None"
        
        # Get current season
        current_season = get_current_season()
        
        return jsonify({
            'success': True,
            'total_recommendations': total_recommendations,
            'unique_users': unique_users,
            'top_crop': top_crop,
            'current_season': current_season
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/crop_distribution', methods=['GET'])
def analytics_crop_distribution():
    """API endpoint for crop distribution data"""
    try:
        timeframe = request.args.get('timeframe', 'all')
        
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        # Prepare date filter based on timeframe
        date_filter = {}
        if timeframe != 'all':
            now = datetime.now()
            if timeframe == 'year':
                date_filter = {'recommendation_date': {'$gte': datetime(now.year - 1, now.month, now.day)}}
            elif timeframe == 'month':
                if now.month == 1:
                    prev_month = datetime(now.year - 1, 12, now.day)
                else:
                    prev_month = datetime(now.year, now.month - 1, now.day)
                date_filter = {'recommendation_date': {'$gte': prev_month}}
            elif timeframe == 'week':
                from datetime import timedelta
                prev_week = now - timedelta(days=7)
                date_filter = {'recommendation_date': {'$gte': prev_week}}
        
        # Build the pipeline
        pipeline = [
            {"$match": date_filter} if date_filter else {"$match": {}},
            {"$group": {"_id": "$outputs.predicted_crop", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        
        results = list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
        
        crops = [result['_id'] for result in results]
        counts = [result['count'] for result in results]
        
        return jsonify({
            'success': True,
            'labels': crops,
            'data': counts
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/province_stats', methods=['GET'])
def analytics_province_stats():
    """API endpoint for province statistics"""
    try:
        crop_filter = request.args.get('crop', 'all')
        
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        match_stage = {}
        if crop_filter != 'all':
            match_stage = {'outputs.predicted_crop': crop_filter}
        
        pipeline = [
            {"$match": match_stage} if match_stage else {"$match": {}},
            {"$group": {"_id": "$inputs.province", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        results = list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
        results = [result for result in results if result['_id'] and result['_id'] != '']
        
        provinces = [result['_id'] for result in results]
        counts = [result['count'] for result in results]
        
        return jsonify({
            'success': True,
            'labels': provinces,
            'data': counts
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/seasonal_trends', methods=['GET'])
def analytics_seasonal_trends():
    """API endpoint for seasonal trends"""
    try:
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        season_pipeline = [
            {"$group": {"_id": "$outputs.season", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        season_results = list(mongo_data['collections']['crop_recommendations'].aggregate(season_pipeline))
        crops_by_season_pipeline = get_seasonal_crop_trends(mongo_data)
        
        return jsonify({
            'success': True,
            'season_distribution': {
                'labels': [result['_id'] for result in season_results],
                'data': [result['count'] for result in season_results]
            },
            'crops_by_season': crops_by_season_pipeline
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/analytics/soil_analysis', methods=['GET'])
def analytics_soil_analysis():
    """API endpoint for soil parameter analysis by province"""
    try:
        parameter = request.args.get('parameter', 'nitrogen')
        
        parameter_mapping = {
            'nitrogen': 'inputs.nitrogen',
            'phosphorus': 'inputs.phosphorus',
            'potassium': 'inputs.potassium',
            'ph': 'inputs.ph'
        }
        
        param_field = parameter_mapping.get(parameter, 'inputs.nitrogen')
        
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
            
        pipeline = [
            {"$match": {param_field: {"$exists": True}}},
            {"$group": {
                "_id": "$inputs.province",
                "avg_value": {"$avg": f"${param_field}"}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        results = list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
        results = [result for result in results if result['_id'] and result['_id'] != '']
        
        provinces = [result['_id'] for result in results]
        values = [round(result['avg_value'], 2) for result in results]
        
        return jsonify({
            'success': True,
            'parameter': parameter,
            'labels': provinces,
            'data': values
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/mongodb_health', methods=['GET'])
def mongodb_health():
    """Check MongoDB connection status"""
    try:
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'status': 'disconnected'})
        
        mongo_data['client'].admin.command('ping')
        
        return jsonify({
            'success': True,
            'status': 'connected',
            'database': 'mudhumeni_db',
            'collections': ['crop_recommendations']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e)
        })

# Enhanced LLM initialization
def initialize_llm():
    """Initialize with OpenAI as primary, Groq as backup"""
    
    # Try OpenAI first (more reliable)
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print("Using OpenAI GPT-4o-mini for better reliability")
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,  # Slightly higher for more natural responses
                max_tokens=1500,  # Increased for more detailed responses
            )
            return llm
    except Exception as e:
        print(f"OpenAI failed: {e}")
    
    # Fallback to Groq
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            print("Falling back to Groq")
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="llama3-70b-8192",
                temperature=0.3,
                max_tokens=4096,
            )
            return llm
    except Exception as e:
        print(f"Groq also failed: {e}")
    
    raise ValueError("No working API keys found")

# Enhanced response formatting
def format_ai_response(response, conversation_stage="general"):
    """Enhanced response formatting for better readability"""
    import re
    
    # Clean up excessive line breaks
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Ensure proper spacing around headers
    response = re.sub(r'(\n)(#{1,3})', r'\1\n\2', response)
    response = re.sub(r'(#{1,3}.*?)(\n)([^\n#])', r'\1\2\n\3', response)
    
    # Improve list formatting
    response = re.sub(r'(\n)(\d+\.)', r'\1\n\2', response)
    response = re.sub(r'(\n)(-)', r'\1\n\2', response)
    
    # Clean up bold formatting
    response = re.sub(r'\*\*\*([^*]+)\*\*\*', r'**\1**', response)
    
    # Add conversational elements based on stage
    if conversation_stage == "greeting":
        if not response.startswith("Hello") and not response.startswith("Hi"):
            response = "Hello! " + response
    
    return response.strip()

# Enhanced system prompt generation
def generate_enhanced_system_prompt(user_id=None, conversation_memory=None):
    """Generate dynamic system prompt based on conversation context"""
    season = get_current_season()
    seasonal_focus = ", ".join(seasonal_crops[season][:5])
    
    user_location = "Southern Africa"
    farming_type = "various types of"
    expertise_level = "beginner"
    
    # Get user preferences
    if user_id and user_id in user_preferences:
        prefs = user_preferences[user_id]
        if prefs.get("location"):
            user_location = prefs["location"]
        if prefs.get("farming_type"):
            farming_type = prefs["farming_type"]
    
    # Get conversation context
    context_summary = ""
    conversation_stage = "general"
    if conversation_memory:
        context_summary = conversation_memory.get_context_summary()
        conversation_stage = conversation_memory.context.conversation_stage
        expertise_level = conversation_memory.context.user_expertise_level
    
    base_prompt = f"""You are Mudhumeni AI, an experienced and friendly farming advisor specifically designed for farmers in Southern Africa. You have deep knowledge of local farming practices, climate conditions, and agricultural challenges unique to this region.

CURRENT CONTEXT:
- Season: {season} (optimal for: {seasonal_focus})
- User Location: {user_location}
- Farming Focus: {farming_type} farming
- User Expertise: {expertise_level}
- Conversation Stage: {conversation_stage}"""

    if context_summary:
        base_prompt += f"\n- Previous Discussion: {context_summary}"

    base_prompt += f"""

PERSONALITY & COMMUNICATION STYLE:
- Be warm, encouraging, and supportive like a knowledgeable mentor
- Use conversational language that feels natural and engaging
- Show genuine interest in the farmer's specific situation
- Acknowledge their challenges and celebrate their successes
- Use "you" and "your farm" to make responses personal
- Ask thoughtful follow-up questions to better understand their needs

RESPONSE GUIDELINES:
- Adapt your language complexity to the user's expertise level
- For beginners: Use simple terms, explain concepts clearly, provide step-by-step guidance
- For experienced farmers: Use more technical language, focus on optimization and advanced techniques
- Always provide practical, actionable advice relevant to Southern African conditions
- Include seasonal considerations and timing recommendations
- Mention specific crops, techniques, or practices common in the region
- When discussing costs or resources, consider the economic realities of small-scale farming

FORMATTING FOR READABILITY:
- Use clear paragraph breaks for different topics
- Use **bold text** for key terms and important points
- Use numbered lists (1., 2., 3.) for sequential steps
- Use bullet points (-) for options or related items
- Use short, scannable paragraphs (2-3 sentences max)
- End responses with helpful follow-up questions or suggestions when appropriate

REGIONAL EXPERTISE:
- Consider drought-resistant crops and water conservation
- Account for variable rainfall patterns and climate challenges
- Suggest affordable, locally available solutions
- Reference local agricultural extension services when relevant
- Consider subsistence and small-scale commercial farming contexts
- Be aware of common pests and diseases in the region

Remember: You're not just providing information—you're building a relationship with farmers and supporting their journey to better agricultural outcomes."""

    return base_prompt

# Enhanced chatbot response function
def enhanced_chatbot_response(user_input, user_id=None):
    """Enhanced chatbot response with better conversation management"""
    global llm
    
    try:
        if not user_input.strip():
            return "I'm here to help with your farming questions. What would you like to know about agriculture in Southern Africa?"
        
        # Get or create conversation memory
        memory_key = f"{user_id}_memory" if user_id else "default_memory"
        if memory_key not in session:
            session[memory_key] = {
                'conversation_history': [],
                'context': {
                    'farming_context': {},
                    'conversation_stage': 'greeting',
                    'user_expertise_level': 'beginner',
                    'key_insights': []
                }
            }
        
        memory_data = session[memory_key]
        memory = EnhancedConversationMemory()
        
        # Restore memory from session
        memory.conversation_history = memory_data.get('conversation_history', [])
        memory.context.farming_context = memory_data['context'].get('farming_context', {})
        memory.context.conversation_stage = memory_data['context'].get('conversation_stage', 'greeting')
        memory.context.user_expertise_level = memory_data['context'].get('user_expertise_level', 'beginner')
        memory.key_insights = memory_data['context'].get('key_insights', [])
        
        # Handle special commands
        if user_input.lower().startswith("set location:"):
            location = user_input[13:].strip()
            if user_id not in user_preferences:
                user_preferences[user_id] = {}
            user_preferences[user_id]["location"] = location
            return f"Perfect! I've noted that you're farming in {location}. This helps me provide more specific advice for your area. What farming challenges or questions do you have for your region?"
        
        if user_input.lower().startswith("set farming:"):
            farming_type = user_input[12:].strip()
            if user_id not in user_preferences:
                user_preferences[user_id] = {}
            user_preferences[user_id]["farming_type"] = farming_type
            return f"Excellent! I understand you're focused on {farming_type} farming. I'll tailor my recommendations accordingly. What specific aspects of {farming_type} would you like to discuss?"
        
        # Handle common greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greeting in user_input.lower() for greeting in greetings):
            memory.context.conversation_stage = 'greeting'
            # Generate personalized greeting based on season and context
            season = get_current_season()
            seasonal_greeting = f"It's {season} season here in Southern Africa"
            if memory.key_insights:
                return f"Hello again! Great to continue our conversation about farming. {seasonal_greeting} - perfect timing to discuss your agricultural plans. How can I help you today?"
            else:
                return f"Hello! I'm Mudhumeni AI, your farming companion for Southern Africa. {seasonal_greeting}, which brings unique opportunities and challenges. Whether you're planning crops, dealing with pests, or optimizing your harvest, I'm here to help. What's on your mind about farming today?"
        
        # Generate enhanced system prompt with conversation context
        system_prompt = generate_enhanced_system_prompt(user_id, memory)
        
        # Prepare conversation context
        recent_context = memory.get_recent_context(num_exchanges=8)
        
        # Construct the prompt
        if recent_context:
            full_prompt = f"{system_prompt}\n\nRECENT CONVERSATION:\n{recent_context}\n\nUser: {user_input}\n\nRespond as Mudhumeni AI, keeping the conversation natural and building on what we've discussed:"
        else:
            full_prompt = f"{system_prompt}\n\nUser: {user_input}\n\nRespond as Mudhumeni AI:"
        
        # Get response from LLM
        response = llm.invoke(full_prompt)
        ai_response = response.content
        
        # Format the response
        formatted_response = format_ai_response(ai_response, memory.context.conversation_stage)
        
        # Update conversation memory
        memory.add_exchange(user_input, formatted_response)
        
        # Save memory back to session
        session[memory_key] = {
            'conversation_history': memory.conversation_history[-20:],  # Keep last 20 exchanges
            'context': {
                'farming_context': memory.context.farming_context,
                'conversation_stage': memory.context.conversation_stage,
                'user_expertise_level': memory.context.user_expertise_level,
                'key_insights': memory.key_insights[-10:]  # Keep last 10 insights
            }
        }
        
        return formatted_response
        
    except requests.exceptions.RequestException as e:
        print(f"Network error in enhanced chatbot response: {str(e)}")
        return "I'm having trouble connecting to my knowledge base right now. While I work on reconnecting, here's what I can suggest: check your internet connection, and I'll be back to help with your farming questions shortly. In the meantime, is there a general farming principle I might be able to help with?"
    
    except Exception as e:
        print(f"Error in enhanced chatbot response: {str(e)}")
        return "I encountered an unexpected issue, but I'm still here to help! Could you try rephrasing your question? I want to make sure I give you the best farming advice possible."

# Validate crop recommendation data
def validate_recommendation_data(data):
    """Validate recommendation data before insertion"""
    required_fields = ['nitrogen', 'phosphorus', 'potassium', 
                      'temperature', 'humidity', 'ph', 'rainfall']
    
    # Check for missing required fields
    for field in required_fields:
        if field not in data['inputs']:
            return False, f"Missing required field: {field}"
            
    # Validate numeric fields are within reasonable ranges
    if not (0 <= data['inputs']['nitrogen'] <= 150):
        return False, "Nitrogen must be between 0 and 150 mg/kg"
        
    if not (0 <= data['inputs']['phosphorus'] <= 150):
        return False, "Phosphorus must be between 0 and 150 mg/kg"
        
    if not (0 <= data['inputs']['potassium'] <= 150):
        return False, "Potassium must be between 0 and 150 mg/kg"
        
    if not (0 <= data['inputs']['temperature'] <= 50):
        return False, "Temperature must be between 0 and 50 °C"
        
    if not (0 <= data['inputs']['humidity'] <= 100):
        return False, "Humidity must be between 0 and 100%"
        
    if not (0 <= data['inputs']['ph'] <= 14):
        return False, "pH must be between 0 and 14"
        
    if not (0 <= data['inputs']['rainfall'] <= 5000):
        return False, "Rainfall must be between 0 and 5000 mm"
        
    return True, "Data is valid"

# Sanitize user input to prevent injection attacks
def sanitize_input(input_string):
    """Remove potentially dangerous characters"""
    if input_string is None:
        return None
    # Remove MongoDB operators and special characters
    sanitized = re.sub(r'[${}()"]', '', input_string)
    return sanitized

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is logged in and is an admin
        if 'user_id' not in session or 'is_admin' not in session or not session['is_admin']:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# MongoDB Analytics Functions (keeping existing functionality)
def get_crop_statistics_by_province(mongo_data):
    """Get crop recommendation statistics grouped by province"""
    try:
        pipeline = [
            {"$group": {
                "_id": {
                    "province": "$inputs.province",
                    "crop": "$outputs.predicted_crop"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$group": {
                "_id": "$_id.province",
                "crops": {
                    "$push": {
                        "crop": "$_id.crop",
                        "count": "$count"
                    }
                },
                "total": {"$sum": "$count"}
            }},
            {"$sort": {"total": -1}}
        ]
        
        return list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
    except Exception as e:
        print(f"Error getting crop statistics: {str(e)}")
        return []

def get_seasonal_crop_trends(mongo_data):
    """Get crop recommendation trends by season"""
    try:
        pipeline = [
            {"$group": {
                "_id": {
                    "season": "$outputs.season",
                    "crop": "$outputs.predicted_crop"
                },
                "count": {"$sum": 1}
            }},
            {"$sort": {"count": -1}},
            {"$group": {
                "_id": "$_id.season",
                "crops": {
                    "$push": {
                        "crop": "$_id.crop",
                        "count": "$count"
                    }
                },
                "total": {"$sum": "$count"}
            }},
            {"$sort": {"_id": 1}}
        ]
        
        return list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
    except Exception as e:
        print(f"Error getting seasonal crop trends: {str(e)}")
        return []

def get_user_recommendation_stats(mongo_data, user_id):
    """Get statistics about a user's recommendations"""
    try:
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": "$outputs.predicted_crop",
                "count": {"$sum": 1},
                "last_recommended": {"$max": "$recommendation_date"}
            }},
            {"$sort": {"count": -1}},
            {"$project": {
                "_id": 0,
                "crop": "$_id",
                "count": 1,
                "last_recommended": 1
            }}
        ]
        
        return list(mongo_data['collections']['crop_recommendations'].aggregate(pipeline))
    except Exception as e:
        print(f"Error getting user stats: {str(e)}")
        return []

# Initialize outside main to be global
print("Initializing Enhanced Mudhumeni AI Chatbot......")
llm = initialize_llm()

# Initialize MongoDB
mongo_data = setup_mongodb()
if mongo_data:
    app.config['MONGO_DATA'] = mongo_data
    print("MongoDB integration initialized successfully")
else:
    print("WARNING: MongoDB connection failed. Recommendation storage will be disabled.")
    app.config['MONGO_DATA'] = None

# Landing page route
@app.route('/')
@app.route('/landing')
def landing():
    return render_template('landing.html')

# Chatbot route
@app.route('/chatbot')
def chatbot():
    # Generate a user ID if not exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # Initialize chat history if needed
    if 'chat_history' not in session:
        session['chat_history'] = []
        
    return render_template('index.html')

# Crop recommendation route
@app.route('/crop-recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Load the model and scalers
        model = pickle.load(open('model.pkl', 'rb'))
        minmaxscaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
        standscaler = pickle.load(open('standscaler.pkl', 'rb'))
        
        # Get data from form
        data = request.form
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Get province from form (add this field to your form)
        province = sanitize_input(data.get('province', ''))
        
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        mx_features = minmaxscaler.transform(features)
        sc_mx_features = standscaler.transform(mx_features)
        prediction = model.predict(sc_mx_features)
        
        # Crop dictionary
        crop_dict = {
            1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
            6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
            11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate',
            15: 'lentil', 16: 'blackgram', 17: 'mungbean', 18: 'mothbeans',
            19: 'pigeonpeas', 20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
        }
        
        # Add Southern African context to the prediction
        predicted_crop = crop_dict[prediction[0]]
        
        # Get current season
        season = get_current_season()
        
        # Add seasonal advice
        seasonal_advice = ""
        if predicted_crop in seasonal_crops[season]:
            seasonal_advice = f"Good choice! {predicted_crop.title()} is well-suited for the current {season} season in Southern Africa."
        else:
            appropriate_season = next((s for s, crops in seasonal_crops.items() if predicted_crop in crops), None)
            if appropriate_season:
                seasonal_advice = f"Note: {predicted_crop.title()} is typically more suited for the {appropriate_season} season in Southern Africa. Consider planning accordingly."
        
        # Create a recommendation document for MongoDB
        user_id = session.get('user_id', str(uuid.uuid4()))
        if 'user_id' not in session:
            session['user_id'] = user_id
            
        recommendation_data = {
            "user_id": user_id,
            "recommendation_date": datetime.now(),
            "inputs": {
                "nitrogen": N,
                "phosphorus": P,
                "potassium": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
                "province": province
            },
            "outputs": {
                "predicted_crop": predicted_crop,
                "season": season,
                "seasonal_advice": seasonal_advice
            }
        }
        
        # Validate recommendation data
        is_valid, message = validate_recommendation_data(recommendation_data)
        if not is_valid:
            return jsonify({'success': False, 'error': message})
        
        # Save to MongoDB if connection exists
        mongo_data = app.config.get('MONGO_DATA')
        if mongo_data:
            try:
                # Insert recommendation to MongoDB
                result = mongo_data['collections']['crop_recommendations'].insert_one(recommendation_data)
                print(f"Recommendation saved with ID: {result.inserted_id}")
            except Exception as e:
                print(f"Error saving recommendation to MongoDB: {str(e)}")
                # Continue even if MongoDB save fails
        
        return jsonify({
            'success': True, 
            'prediction': predicted_crop,
            'season': season,
            'seasonal_advice': seasonal_advice
        })
    
    except Exception as e:
        print(f"Error in predict_crop: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Route to display recommendation history page
@app.route('/recommendation-history')
def recommendation_history_page():
    # Ensure user ID exists
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        
    return render_template('recommendation_history.html')

# API endpoint to get recommendation history for the current user
@app.route('/api/recommendation_history', methods=['GET'])
def get_recommendation_history():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID not found in session'})
        
        # Get MongoDB connection
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        # Query recommendations for this user, sort by date descending (newest first)
        cursor = mongo_data['collections']['crop_recommendations'].find(
            {"user_id": user_id}
        ).sort("recommendation_date", -1)
        
        # Convert cursor to list and format dates
        recommendations = []
        for rec in cursor:
            # Convert ObjectId to string and datetime to ISO format for JSON serialization
            rec['_id'] = str(rec['_id'])
            rec['recommendation_date'] = rec['recommendation_date'].isoformat()
            recommendations.append(rec)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error retrieving recommendation history: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# API endpoint to get details for a specific recommendation
@app.route('/api/recommendation/<recommendation_id>', methods=['GET'])
def get_recommendation_detail(recommendation_id):
    try:
        from bson.objectid import ObjectId
        
        # Get MongoDB connection
        mongo_data = app.config.get('MONGO_DATA')
        if not mongo_data:
            return jsonify({'success': False, 'error': 'Database connection not available'})
        
        # Query specific recommendation
        recommendation = mongo_data['collections']['crop_recommendations'].find_one(
            {"_id": ObjectId(recommendation_id)}
        )
        
        if not recommendation:
            return jsonify({'success': False, 'error': 'Recommendation not found'})
        
        # Convert ObjectId to string and datetime to ISO format for JSON serialization
        recommendation['_id'] = str(recommendation['_id'])
        recommendation['recommendation_date'] = recommendation['recommendation_date'].isoformat()
        
        return jsonify({
            'success': True,
            'recommendation': recommendation
        })
    except Exception as e:
        print(f"Error retrieving recommendation detail: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form["user_input"]
    user_id = session.get('user_id')
    
    # Generate user ID if not exists
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
    
    # Get existing chat history or initialize
    chat_history = session.get('chat_history', [])
    
    # Add user input to history
    chat_history.append({"role": "user", "content": user_input})
    
    # Get enhanced response from chatbot
    response = enhanced_chatbot_response(user_input, user_id)
    
    # Add response to history
    chat_history.append({"role": "assistant", "content": response})
    
    # Keep history manageable (last 30 exchanges)
    if len(chat_history) > 60:  # 30 user + 30 assistant messages
        chat_history = chat_history[-60:]
    
    # Save updated history to session
    session['chat_history'] = chat_history
    
    return jsonify({"response": response, "chat_history": chat_history})

# About route
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    # Load ML model and scalers
    try:
        # Load the prediction model
        model = pickle.load(open('model.pkl', 'rb'))
        minmaxscaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
        standscaler = pickle.load(open('standscaler.pkl', 'rb'))
        print("ML models loaded successfully")
    except Exception as e:
        print(f"Error loading ML models: {str(e)}")
        print("Application will run, but crop recommendation may not work properly")
    
    # Get port from environment variable for deployment platforms like Render
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app with production settings
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )