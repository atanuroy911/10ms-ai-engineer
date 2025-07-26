"""
Test script for Bengali-English RAG Flask API
Run this script to test the API endpoints
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return data.get('rag_initialized', False)
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the Flask app is running.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_language_detection():
    """Test language detection endpoint"""
    print("\n🔍 Testing language detection...")
    
    test_cases = [
        {"text": "অনুপমের বন্ধু হরিশ কোথায় কাজ করে?", "expected": "bengali"},
        {"text": "Where does Harish work?", "expected": "english"},
        {"text": "আমি ভালো আছি", "expected": "bengali"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/api/detect_language",
                headers=HEADERS,
                json={"text": test_case["text"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                detected = data.get('detected_language')
                expected = test_case['expected']
                status = "✅" if detected == expected else "⚠️"
                print(f"  {status} Test {i}: '{test_case['text'][:30]}...' -> {detected}")
            else:
                print(f"  ❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Test {i} error: {e}")

def test_translation():
    """Test translation endpoint"""
    print("\n🔍 Testing translation...")
    
    test_cases = [
        {"text": "অনুপমের বন্ধু হরিশ কোথায় কাজ করে?", "from_lang": "Bengali"},
        {"text": "আমি ভালো আছি", "from_lang": "Bengali"},
        {"text": "Hello, how are you?", "from_lang": "English"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/api/translate",
                headers=HEADERS,
                json={"text": test_case["text"]}
            )
            
            if response.status_code == 200:
                data = response.json()
                original = data.get('original_text', '')
                translated = data.get('translated_text', '')
                lang = data.get('original_language', '')
                confidence = data.get('translation_confidence', 0)
                
                print(f"  ✅ Test {i} ({test_case['from_lang']}):")
                print(f"     Original: {original}")
                print(f"     Translated: {translated}")
                print(f"     Confidence: {confidence:.2f}")
            else:
                print(f"  ❌ Test {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Test {i} error: {e}")

def test_question_answering():
    """Test question answering endpoint"""
    print("\n🔍 Testing question answering...")
    
    test_cases = [
        {
            "question": "অনুপমের বন্ধু হরিশ কোথায় কাজ করে?",
            "language": "Bengali"
        },
        {
            "question": "Where does Anupam's friend Harish work?",
            "language": "English"
        },
        {
            "question": "বিবাহ ভাঙার পর হতে কল্যাণী কোন ব্রত গ্রহণ করে?",
            "language": "Bengali"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  📝 Test {i} ({test_case['language']}):")
        print(f"     Question: {test_case['question']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/ask",
                headers=HEADERS,
                json={
                    "question": test_case["question"],
                    "show_translation_details": True
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    answer = data.get('answer', '')
                    query_lang = data.get('query_language', '')
                    answer_lang = data.get('answer_language', '')
                    context_count = len(data.get('context', []))
                    
                    print(f"     ✅ Success!")
                    print(f"     Answer: {answer[:100]}...")
                    print(f"     Query Language: {query_lang}")
                    print(f"     Answer Language: {answer_lang}")
                    print(f"     Context Sources: {context_count}")
                    
                    # Show translation details if available
                    if data.get('translation_details'):
                        trans_details = data['translation_details']
                        if trans_details.get('original_language') == 'bengali':
                            print(f"     English Query: {trans_details.get('translated_text', '')}")
                else:
                    print(f"     ❌ Failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"     ❌ HTTP Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"     Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"     Error: {response.text}")
                    
        except Exception as e:
            print(f"     ❌ Exception: {e}")

def test_invalid_requests():
    """Test error handling with invalid requests"""
    print("\n🔍 Testing error handling...")
    
    # Test empty question
    try:
        response = requests.post(
            f"{BASE_URL}/api/ask",
            headers=HEADERS,
            json={"question": ""}
        )
        
        if response.status_code == 400:
            print("  ✅ Empty question properly rejected")
        else:
            print(f"  ⚠️ Empty question not properly handled: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error testing empty question: {e}")
    
    # Test missing question field
    try:
        response = requests.post(
            f"{BASE_URL}/api/ask",
            headers=HEADERS,
            json={}
        )
        
        if response.status_code == 400:
            print("  ✅ Missing question field properly rejected")
        else:
            print(f"  ⚠️ Missing question field not properly handled: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error testing missing question: {e}")

def main():
    """Run all tests"""
    print("🧪 Bengali-English RAG API Test Suite")
    print("=" * 50)
    
    # Check if the server is running
    rag_initialized = test_health_check()
    
    if not rag_initialized:
        print("\n⚠️ RAG system not initialized or server not running.")
        print("Please make sure:")
        print("1. Flask app is running (python app.py)")
        print("2. Vector store exists (run the notebook first)")
        print("3. Environment variables are set correctly")
        return
    
    # Run all tests
    test_language_detection()
    test_translation()
    test_question_answering()
    test_invalid_requests()
    
    print("\n" + "=" * 50)
    print("🏁 Test suite completed!")
    print("\nIf you see ❌ errors, check:")
    print("- Flask app logs for detailed error messages")
    print("- OpenAI API key and credits")
    print("- Vector store availability")
    print("- Internet connection for translation services")

if __name__ == "__main__":
    main()
