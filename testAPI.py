import requests
import time
import sys
from colorama import Fore, Style, init

init(autoreset=True)

def type_print(text, delay=0.015, color=Fore.GREEN):
    sys.stdout.write(f"{color}") 
    
    cleaned_text = text.strip().strip('"').strip('}').strip('{')
    
    for character in cleaned_text:
        sys.stdout.write(character)
        sys.stdout.flush()         
        time.sleep(delay)
        
    sys.stdout.write(f"{Style.RESET_ALL}") 
    print() 

def print_section_header(title, color=Fore.YELLOW):
    print(f"\n{color}{'='*60}")
    print(f"{color}>>> {title} <<<")
    print(f"{color}{'='*60}{Style.RESET_ALL}")

def print_query(courseId, query):
    print(f"{Fore.BLUE}{Style.BRIGHT}USER [{courseId}] >{Style.RESET_ALL} {query}")
    sys.stdout.write(f"{Fore.CYAN}{Style.BRIGHT}AI ANSWER > {Style.RESET_ALL}")
    sys.stdout.flush()

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:5001"

# NOTE: Update these paths to be valid on the machine running the script
nlp_files = [
    "DemoDoc/NLP/Lec 3 Preprocessing.pdf",
    "DemoDoc/NLP/Ethics in NLP.pdf"
]

fyp_files = [
    "DemoDoc/NIS/11-Digital-Certificates.pdf"
]

# --- CORE INTERACTIVE FUNCTION ---
def interactive_chat_mode(courseId):
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}--- Entering Interactive Chat Mode for {courseId} ---{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Ask questions about the course content. Type 'quit' or 'exit' to return to the main menu.")

    while True:
        try:
            user_input = input(f"{Fore.BLUE}{Style.BRIGHT}USER [{courseId}] > {Style.RESET_ALL}")
            
            if user_input.lower() in ['quit', 'exit']:
                break

            ask_payload = {"courseId": courseId, "question": user_input}
            
            sys.stdout.write(f"{Fore.CYAN}{Style.BRIGHT}AI ANSWER > {Style.RESET_ALL}")
            sys.stdout.flush()

            resp = requests.post(f"{API_URL}/ask", json=ask_payload)
            data = resp.json()

            if resp.status_code == 200:
                type_print(data.get('answer', 'Error'), delay=0.015, color=Fore.GREEN)
            else:
                type_print(data.get('message', 'API Error'), delay=0.03, color=Fore.RED)

        except Exception as e:
            print(f"{Fore.RED}CONNECTION ERROR: Could not reach Flask API. Details: {e}{Style.RESET_ALL}")
            break

def run_test():
    print(f"{Fore.GREEN}{Style.BRIGHT}\n[ RAG MICROSERVICE TEST SEQUENCE STARTED ]{Style.RESET_ALL}")
    
    # ----------------------------------
    # 1 & 2. INDEXING (Initial Setup)
    # ----------------------------------
    print_section_header("1. INDEXING SETUP: BUILDING KNOWLEDGE BASE", Fore.BLUE)
    
    # Index NLP
    print(f"Indexing NLP_CT-485...")
    requests.post(f"{API_URL}/update_course", data={"courseId": "NLP_CT-485", "pdfPaths": ",".join(nlp_files)})
    
    # Index NIS
    print(f"Indexing NIS_CT-486...")
    requests.post(f"{API_URL}/update_course", data={"courseId": "NIS_CT-486", "pdfPaths": ",".join(fyp_files)})
    
    print(f"{Fore.GREEN}Indexing Complete! Ready for interactive Q&A.{Style.RESET_ALL}")

    # ----------------------------------
    # 3. INTERACTIVE CHAT MODE
    # ----------------------------------
   
    while True:
        print_section_header("2. INTERACTIVE DEMO SELECTION", Fore.YELLOW)
        course_choice = input(f"Enter Course ID to Chat (e.g., {Fore.GREEN}NIS_CT-486{Style.RESET_ALL} or {Fore.GREEN}NLP_CT-485{Style.RESET_ALL}) or type 'end': ")
        
        if course_choice.lower() == 'end':
            break
        
        if course_choice in ["NIS_CT-486", "NLP_CT-485"]:
            interactive_chat_mode(course_choice)
        else:
            print(f"{Fore.RED}Invalid Course ID. Please try again.{Style.RESET_ALL}")
            
    print(f"{Fore.GREEN}{Style.BRIGHT}\n[  RAG MICROSERVICE DEMONSTRATION ENDED ]{Style.RESET_ALL}\n")

if __name__ == "__main__":
    run_test()