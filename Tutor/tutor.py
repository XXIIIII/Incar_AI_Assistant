"""
AI Travel Tutor Module for Incar AI Assistant

This module provides intelligent travel planning and guidance using LangChain and Google Gemini AI.
Features include:
- Personalized travel itinerary generation
- Multi-format response (speech, text, table, detailed guides)
- Markdown to speech normalization for TTS compatibility
- Comprehensive travel advice with transportation and logistics
"""

import os
import re
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from .search import web_search


def normalize_markdown_for_tts(markdown_text):
    """
    Convert markdown text to plain text suitable for text-to-speech.
    
    Removes all markdown formatting while preserving the essential content
    for natural speech synthesis.
    
    Args:
        markdown_text (str): Input text with markdown formatting
        
    Returns:
        str: Clean text suitable for TTS
    """
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', markdown_text)
    
    # Convert headers to plain text (remove '#' symbols)
    clean_text = re.sub(r'#+\s*(.*?)\s*#*', r'\1', clean_text)
    
    # Remove image alt text and links (keep the descriptive part)
    clean_text = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', clean_text)
    clean_text = re.sub(r'\[[^\]]*\]\([^)]*\)', '', clean_text)
    
    # Remove inline formatting (bold, italic)
    clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'_([^_]+)_', r'\1', clean_text)
    clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
    
    # Convert lists to plain text
    clean_text = re.sub(r'^\s*[-*]\s+', '', clean_text, flags=re.MULTILINE)
    
    # Remove blockquotes
    clean_text = re.sub(r'^\s*>+\s+', '', clean_text, flags=re.MULTILINE)
    
    # Remove code blocks and inline code
    clean_text = re.sub(r'```.*?```', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'`.*?`', '', clean_text)
    
    # Remove horizontal rules
    clean_text = re.sub(r'^---+$', '', clean_text, flags=re.MULTILINE)
    
    # Normalize whitespace to a single space
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Trim leading and trailing whitespace
    clean_text = clean_text.strip()
    
    return clean_text


def save_response_to_files(result: dict, prompt_messages, format_type=None):
    """
    Save AI responses and prompts to markdown files for debugging and analysis.
    
    Args:
        result (dict): AI response content by format type
        prompt_messages (list): List of message pairs used for generation
        format_type (str): Optional format identifier for filename
    """
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%m%d_%H_%M_%S")
    
    # Create session directory
    session_dir = f'output/Tutor_{format_type if format_type else "default"}_{timestamp}'
    os.makedirs(session_dir, exist_ok=True)
    
    # Save AI responses
    response_path = os.path.join(session_dir, 'respond.md')
    with open(response_path, 'w', encoding='utf-8') as file:
        for key, value in result.items():
            file.write('=' * 15 + f' {key.upper()} ' + '=' * 15 + '\n')
            file.write(str(value))
            file.write('\n\n')
    
    # Save prompts for debugging
    prompt_path = os.path.join(session_dir, 'prompt.txt')
    with open(prompt_path, 'w', encoding='utf-8') as file:
        for i, (system_msg, human_msg) in enumerate(prompt_messages):
            file.write('=' * 15 + f' PROMPT {i+1} - SYSTEM ' + '=' * 15 + '\n')
            file.write('\n'.join(system_msg.content))
            file.write('\n\n' + '=' * 15 + f' PROMPT {i+1} - HUMAN ' + '=' * 15 + '\n')
            file.write('\n'.join(human_msg.content))
            file.write('\n\n')
    
    print(f"Output saved to {session_dir}/")


class Tutor:
    """
    AI Travel Tutor powered by Google Gemini and LangChain.
    
    Provides intelligent travel planning with multiple output formats:
    - Speech: Conversational responses for voice interaction
    - Table: Structured itinerary in markdown table format  
    - Text: Detailed travel descriptions
    - Tutor: Educational content about destinations
    """
    
    def __init__(self, model="gemini-1.5-pro-latest"):
        """
        Initialize the AI Tutor with required API keys and model.
        
        Args:
            model (str): Google Gemini model identifier
            
        Raises:
            ValueError: If required environment variables are not set
        """
        self.model = model
        
        # Validate required environment variables
        if not os.environ.get("GOOGLE_GEMINI_KEY"):
            raise ValueError(
                "GOOGLE_GEMINI_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        
        # Initialize reference examples for prompt engineering
        self._initialize_reference_examples()
        
        print("AI Travel Tutor initialized successfully")
    
    def invoke(self, user_message, response_format=None, save_output=True):
        """
        Generate AI response based on user input and desired format.
        
        Args:
            user_message (str): User's travel query or request
            response_format (str): Desired output format ('speech', 'table', 'text', 'tutor')
            save_output (bool): Whether to save responses to files
            
        Returns:
            dict: AI-generated responses by format type
        """
        # Initialize the AI model
        model = ChatGoogleGenerativeAI(
            model=self.model, 
            google_api_key=os.environ.get("GOOGLE_GEMINI_KEY")
        )
        
        print("Generating AI response...")
        
        # Get web search data (currently disabled for stability)
        web_context = ''  # web_search(user_message) if needed
        
        prompt_history = []
        
        if not response_format:
            # Default behavior: Generate both speech and table formats
            print("Generating speech response...")
            speech_prompt = self._create_prompt(user_message, web_context, 'speech')
            speech_result = model.invoke(speech_prompt)
            
            # Normalize for TTS compatibility
            speech_content = normalize_markdown_for_tts(speech_result.content)
            
            print("Generating table response...")
            table_request = f"請幫我整理以下行程，並以表格方式呈現：{speech_content}"
            table_prompt = self._create_prompt(table_request, web_context, 'table')
            table_result = model.invoke(table_prompt)
            
            response = {
                "speech": speech_content,
                "table": table_result.content
            }
            prompt_history.extend([speech_prompt, table_prompt])
            
        else:
            # Generate specific format
            print(f"Generating {response_format} response...")
            prompt = self._create_prompt(user_message, web_context, response_format)
            result = model.invoke(prompt)
            
            response = {
                response_format: result.content
            }
            prompt_history.append(prompt)
        
        # Save outputs for debugging and analysis
        if save_output:
            save_response_to_files(response, prompt_history, response_format)
        
        print("Response generation completed")
        return response
    
    def _create_prompt(self, user_message, web_context, output_format=None):
        """
        Create structured prompt messages for the AI model.
        
        Args:
            user_message (str): User's input message
            web_context (str): Additional web search context
            output_format (str): Desired response format
            
        Returns:
            list: SystemMessage and HumanMessage pair
        """
        # Base system instructions
        system_instructions = [
            "你現在是一個專業的旅遊導遊和行程規劃師",
            "請根據旅客的需求，提供詳細的行程安排、景點推薦或景點介紹",
            "請參考範例格式，並根據實際地點和時間調整相關資訊",
            "在規劃行程時，請包含交通資訊、停車建議、注意事項等實用資訊",
            "請使用繁體中文回覆，回答要準確且實用，避免使用表情符號",
            "以下為相關參考資訊:"
        ]
        
        # Add web context if available
        if web_context:
            system_instructions.append(web_context)
        
        # Format-specific instructions
        user_instructions = []
        
        if output_format == 'speech':
            user_instructions.extend([
                "請以自然對話的方式回答，適合語音播報",
                "回答要簡潔明瞭，建議在100字內完成",
                "重點說明推薦景點和選擇理由",
                "考慮景點間距離和交通時間，確保行程合理"
            ])
        else:
            # Add reference examples for non-speech formats
            system_instructions.extend([
                "請使用繁體中文和適當的Markdown格式回答",
                f"參考以下範例格式：\n表格格式：{self.table_reference}\n"
                f"文字格式：{self.text_reference}\n景點介紹：{self.guide_reference}"
            ])
            
            if output_format == "table":
                user_instructions.extend([
                    f"請以表格方式呈現行程安排，參考格式：{self.table_reference}",
                    "表格須包含：時間、活動、地點、停留時間、交通時間、備註",
                    "考慮實際交通狀況和景點開放時間"
                ])
            elif output_format == "text":
                user_instructions.extend([
                    f"請以文字方式詳細描述行程，參考格式：{self.text_reference}",
                    "包含景點介紹、活動安排、交通方式等完整資訊"
                ])
            elif output_format == "tutor":
                user_instructions.extend([
                    f"請提供詳細的景點介紹和歷史背景，參考格式：{self.guide_reference}",
                    "包含景點特色、歷史文化、參觀重點等教育性內容"
                ])
        
        # Add user's original message
        user_instructions.append(user_message)
        
        return [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_instructions)
        ]
    
    def _initialize_reference_examples(self):
        """Initialize reference examples for different output formats."""
        
        self.table_reference = """### 台北一日遊自駕行程表

| 時間          | 活動                       | 地點                         | 預計停留時間 | 預計行車時間 | 備註                      |
|-------------|--------------------------|----------------------------|------------|------------|-------------------------|
| 09:00 - 09:30 | 出發及前往第一個景點            | 自住宿出發                     | -          | 30分鐘       | 請確認車輛油量充足               |
| 09:30 - 11:00 | 參觀國立故宮博物院              | 國立故宮博物院                   | 1.5小時       | -          | 需預留時間購票及安檢             |
| 11:00 - 11:45 | 前往陽明山國家公園              | 從故宮開車至陽明山                | -          | 45分鐘       | 可能因山路而耗時較多             |
| 11:45 - 14:00 | 遊覽陽明山及午餐               | 陽明山國家公園                   | 2.25小時      | -          | 可在山上餐廳享用午餐            |
| 14:00 - 14:40 | 前往士林官邸公園               | 從陽明山前往士林官邸             | -          | 40分鐘       | 注意山路駕駛                    |
| 14:40 - 16:10 | 參觀士林官邸公園及花園           | 士林官邸公園                     | 1.5小時       | -          | 可以享受一段輕鬆的散步時間         |
| 16:10 - 16:40 | 前往台北101                  | 從士林官邸開車至台北101           | -          | 30分鐘       | 途中可能遭遇下午的車流高峰        |
| 16:40 - 18:10 | 探索台北101及購物                | 台北101                       | 1.5小時       | -          | 如需上觀景台，建議提前網上購票       |
| 18:10 - 18:40 | 前往晚餐地點                  | 從台北101前往晚餐地點              | -          | 30分鐘       | 可選擇在信義區內尋找美食          |
| 18:40 - 20:40 | 晚餐及自由活動                | 信義區美食餐廳                    | 2小時         | -          | 推薦試試當地特色料理              |
| 20:40        | 返回住宿地點                 | 返回住宿                       | -          | -          | 今日行程結束                   |"""

        self.text_reference = """這是為您安排的台北自駕一日遊行程。

**台北自駕一日遊行程**

**09:00** - 從住宿地點出發，前往國立故宮博物院。

**09:30-11:00** - 參觀國立故宮博物院。這裡收藏有豐富的中國古代藝術品，是瞭解中國歷史和文化的絕佳場所。

**11:00-11:45** - 駕車前往陽明山國家公園。

**11:45-14:00** - 在陽明山國家公園遊覽和午餐。公園以其美麗的自然風景和多樣的地質現象著稱，是台北近郊最受歡迎的自然景點之一。

**14:00-14:40** - 駕車前往士林官邸公園。

**14:40-16:10** - 參觀士林官邸公園。這裡曾是前總統蔣中正和宋美齡的官邸，現在是一個公共花園，內有美麗的花園和散步小徑。

**16:10-16:40** - 駕車前往台北101。

**16:40-18:10** - 探索台北101。您可以到觀景台欣賞台北市景，或在購物中心購物。

**18:10-18:40** - 駕車前往晚餐地點。

**18:40-20:40** - 在信義區享用晚餐，這裡有許多提供當地及國際美食的餐廳。

**20:40** - 返回住宿地點，結束充實的一天。

### 景點介紹

- **國立故宮博物院**：擁有超過69萬件古代中國藝術品的國立故宮博物院是探索中國豐富文化歷史的寶庫。
- **陽明山國家公園**：陽明山是台北市的後花園，以其溫泉、山峰和豐富的植被而聞名，是自然愛好者的天堂。
- **士林官邸公園**：這個具有歷史意義的地方不僅提供了一個美麗的公園空間，還能讓遊客一窺台灣近代政治歷史。
- **台北101**：這座前世界最高建築不僅是台北市的象徵，其內部的購物中心和觀景台也是來台必訪的地點之一。

希望這個行程能夠讓您的台北之旅增添不少樂趣，讓您的自駕遊充滿難忘的經歷和美好的回憶！"""

        self.guide_reference = """當然可以！以下是您行程中每個景點的介紹以及其背景歷史：

### 1. 國立故宮博物院 (National Palace Museum)
- **介紹**：國立故宮博物院擁有世界上最豐富的中國古代藝術品收藏，其藏品大部分來自中國大陸清朝故宮，在中國內戰期間運往台灣。這些藏品涵蓋了書畫、陶瓷、玉器等各類精美藝術品。
- **歷史**：故宮博物院原本位於北京，1949年隨著國民政府撤退到台灣，許多寶貴文物也被運來台灣以避免戰亂破壞。現在的國立故宮博物院於1965年在台北重新開放，成為展示中國五千年文化遺產的重要場所。

### 2. 陽明山國家公園 (Yangmingshan National Park)
- **介紹**：陽明山國家公園以其豐富的自然景觀著稱，包括溫泉、瀑布、各種植物與活火山地質景觀。公園內有多條健行路徑，是觀賞櫻花和賞鳥的絕佳地點。
- **歷史**：陽明山的名稱來自於明代哲學家王陽明，公園地區在日治時期開始成為度假地，當時稱為「草山」。1950年，為紀念王陽明而改稱為陽明山。此後，該地區逐步發展成為國家公園，並於1985年正式命名為陽明山國家公園。

### 3. 士林官邸公園 (Shilin Official Residence)
- **介紹**：士林官邸是前中華民國總統蔣中正和夫人宋美齡的官邸，現在對外開放為公園，園區內有美麗的花園和蔣介石的歷史展覽。
- **歷史**：官邸建於1950年，原為蔣中正的私人住所和公務活動場所。2000年後，官邸和其周圍花園被開放給公眾參觀，成為了一個集歷史與文化於一身的旅遊景點。

### 4. 台北101 (Taipei 101)
- **介紹**：台北101曾是世界上最高的建築，樓高101層，是台北市的標誌性建築。內部除了有高端購物中心、辦公空間，還有一個觀景台提供壯觀的城市景觀。
- **歷史**：台北101於2004年完工，當時超越馬來西亞的吉隆坡雙塔成為世界最高建築。建築設計融合了傳統中國元素與現代技術，其中最著名的是大樓頂部的巨大防震球，能夠抵抗台灣頻繁的地震。

以上是您行程中景點的介紹和相關歷史背景，希望這些信息可以讓您對這些地方有更深的了解和期待。祝您在台北的一日遊充滿愉快的發現和美好的體驗！"""


# Example usage for testing
if __name__ == "__main__":
    # Test the tutor functionality
    try:
        tutor = Tutor()
        response = tutor.invoke("我想要規劃一個台北一日遊的行程", response_format="speech")
        print("Speech Response:", response.get("speech", "No speech response"))
    except Exception as e:
        print(f"Error testing tutor: {e}")