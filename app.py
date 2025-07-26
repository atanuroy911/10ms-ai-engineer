"""
Flask Application for Bengali-English Translation Augmented RAG System
Provides REST API and Web UI for multilingual document querying
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import time
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Validate configuration
try:
    Config.validate_config()
    logger.info("✅ Configuration validated successfully")
except ValueError as e:
    logger.error(f"❌ Configuration error: {e}")
    logger.error("Please check your .env file or environment variables")
    exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY

class TranslationAugmentedRAG:
    """
    Translation Augmented RAG system for Bengali-English multilingual queries
    """
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.chat_model = Config.CHAT_MODEL
        self.vector_store_path = Config.VECTOR_STORE_PATH
        self.retrieval_k = Config.RETRIEVAL_K
        self.retrieval_score_threshold = Config.RETRIEVAL_SCORE_THRESHOLD
        self.translation_rate_limit = Config.TRANSLATION_RATE_LIMIT
        self.chain = None
        self._initialize_rag_chain()
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text"""
        try:
            if not text.strip():
                return 'unknown'
            
            detected = detect(text)
            
            if detected == 'bn':
                return 'bengali'
            elif detected == 'en':
                return 'english'
            else:
                # Check for Bengali Unicode characters
                bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
                total_chars = len([char for char in text if char.isalpha()])
                
                if total_chars > 0 and (bengali_chars / total_chars) > 0.3:
                    return 'bengali'
                else:
                    return 'english'
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            if bengali_chars > 5:
                return 'bengali'
            return 'english'

    def translate_text_to_english(self, text: str) -> Dict:
        """Translate Bengali text to English and return metadata"""
        if not text.strip():
            return {
                'original_text': text,
                'translated_text': text,
                'original_language': 'unknown',
                'translation_confidence': 0.0,
                'translation_method': 'none'
            }
        
        detected_lang = self.detect_language(text)
        
        if detected_lang == 'english':
            return {
                'original_text': text,
                'translated_text': text,
                'original_language': 'english',
                'translation_confidence': 1.0,
                'translation_method': 'none'
            }
        
        try:
            time.sleep(self.translation_rate_limit)  # Rate limiting from config
            translator = GoogleTranslator(source='bn', target='en')
            translated_text = translator.translate(text)
            
            return {
                'original_text': text,
                'translated_text': translated_text,
                'original_language': 'bengali',
                'translation_confidence': Config.DEFAULT_TRANSLATION_CONFIDENCE,
                'translation_method': 'deep_translator'
            }
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                'original_text': text,
                'translated_text': text,
                'original_language': detected_lang,
                'translation_confidence': 0.0,
                'translation_method': 'failed'
            }

    def _initialize_rag_chain(self):
        """Initialize the RAG chain with multilingual support"""
        try:
            # Use configurable GPT model for better multilingual support
            model = ChatOpenAI(
                model=self.chat_model,
                temperature=0.2,
                max_tokens=1500
            )
            
            # Multilingual prompt template
            prompt = PromptTemplate.from_template(
                """
                You are a helpful multilingual assistant. You have access to context that was originally in Bengali but has been translated to English for processing.
                
                IMPORTANT INSTRUCTIONS:
                1. Answer based only on the provided context
                2. If the user asks in Bengali, respond in Bengali
                3. If the user asks in English, respond in English  
                4. If you don't know the answer, say "No context available for this question" in the same language as the question
                5. The context provided is English translations of originally Bengali content
                
                User Question: {input}
                Context (English translations): {context}
                
                Answer (respond in the same language as the question):
                """
            )
            
            # Load vector store with English translations
            logger.info(f"Loading English vector store with OpenAI embeddings ({self.embedding_model})...")
            embedding = OpenAIEmbeddings(model=self.embedding_model)
            
            # Check if vector store exists
            if not os.path.exists(self.vector_store_path):
                logger.error(f"Vector store not found at {self.vector_store_path}")
                raise FileNotFoundError(f"Vector store not found. Please run the document ingestion process first.")
            
            vector_store = Chroma(
                persist_directory=self.vector_store_path, 
                embedding_function=embedding
            )

            # Create retriever with configurable parameters
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.retrieval_k,
                    "score_threshold": self.retrieval_score_threshold,
                },
            )

            document_chain = create_stuff_documents_chain(model, prompt)
            self.chain = create_retrieval_chain(retriever, document_chain)
            
            logger.info("RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            self.chain = None

    def ask_question(self, query: str, show_translation_details: bool = True) -> Dict:
        """Ask questions in Bengali or English using the translated vector store"""
        
        if not self.chain:
            return {
                'error': 'RAG system not initialized. Please check vector store availability.',
                'success': False
            }
        
        try:
            # Detect and translate query if needed
            query_translation = self.translate_text_to_english(query)
            original_language = query_translation['original_language']
            english_query = query_translation['translated_text']
            
            # Search vector store with translated query
            result = self.chain.invoke({"input": query})
            answer = result["answer"]
            
            # Prepare context information
            context_info = []
            for doc in result["context"]:
                context_info.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'original_language': doc.metadata.get('original_language', 'Unknown'),
                    'translation_confidence': doc.metadata.get('translation_confidence', 0.0),
                    'content_preview': doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                })
            
            return {
                'success': True,
                'original_query': query,
                'english_query': english_query,
                'query_language': original_language,
                'answer': answer,
                'answer_language': self.detect_language(answer),
                'context': context_info,
                'translation_details': query_translation if show_translation_details else None,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                'error': f'Error processing question: {str(e)}',
                'success': False,
                'timestamp': datetime.now().isoformat()
            }

# Initialize RAG system
rag_system = TranslationAugmentedRAG()

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_system.chain is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """REST API endpoint for asking questions"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Question is required',
                'success': False
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'error': 'Question cannot be empty',
                'success': False
            }), 400
        
        show_details = data.get('show_translation_details', True)
        
        # Process the question
        response = rag_system.ask_question(question, show_details)
        
        if response.get('success', False):
            return jsonify(response)
        else:
            return jsonify(response), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate_text():
    """REST API endpoint for text translation"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Text is required',
                'success': False
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Text cannot be empty',
                'success': False
            }), 400
        
        # Translate the text
        translation_result = rag_system.translate_text_to_english(text)
        translation_result['success'] = True
        translation_result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(translation_result)
            
    except Exception as e:
        logger.error(f"Translation API error: {e}")
        return jsonify({
            'error': f'Translation error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/detect_language', methods=['POST'])
def detect_language_api():
    """REST API endpoint for language detection"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Text is required',
                'success': False
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Text cannot be empty',
                'success': False
            }), 400
        
        # Detect language
        detected_language = rag_system.detect_language(text)
        
        return jsonify({
            'text': text,
            'detected_language': detected_language,
            'success': True,
            'timestamp': datetime.now().isoformat()
        })
            
    except Exception as e:
        logger.error(f"Language detection API error: {e}")
        return jsonify({
            'error': f'Language detection error: {str(e)}',
            'success': False,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Bengali-English Translation Augmented RAG Flask App...")
    print("📚 Features:")
    print("   - Multilingual chat interface (Bengali/English)")
    print("   - REST API for programmatic access")
    print("   - Translation and language detection")
    print("   - Modern TailwindCSS UI")
    print("\n🌐 Endpoints:")
    print("   - GET  /                 - Chat interface")
    print("   - GET  /api/health       - Health check")
    print("   - POST /api/ask          - Ask questions")
    print("   - POST /api/translate    - Translate text")
    print("   - POST /api/detect_language - Detect language")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
