export interface PromptSentence {
  text: string;
  emotion: string;
  translation?: string;
}

export interface CategoryGroup {
  name: string;
  sentences: PromptSentence[];
}

export interface LanguageSheet {
  name: string;
  flag: string;
  value: string;
  categories: CategoryGroup[];
}

export const SENTENCE_DICTS: LanguageSheet[] = [
  {
    name: "English",
    flag: "🇺🇸",
    value: "en",
    categories: [
      {
        name: "🏠 Smart Home & IoT",
        sentences: [
          { text: "Turn off all the lights in the living room and lock the front door.", emotion: "neutral" },
          { text: "Set the kitchen thermostat to seventy-two degrees Fahrenheit.", emotion: "neutral" },
          { text: "Dim the bedroom lamps by fifty percent and play some relaxing music.", emotion: "calm" },
          { text: "Is the garage door closed or did I leave it open?", emotion: "concerned" },
          { text: "Set a timer for fifteen minutes on the smart microwave.", emotion: "neutral" },
          { text: "Turn on the garden sprinkler system for twenty minutes tomorrow morning.", emotion: "neutral" },
          { text: "Please announce that dinner is ready on all smart speaker displays.", emotion: "cheerful" },
          { text: "Show me the live security camera feed from the back patio.", emotion: "concerned" },
          { text: "Increase the fan speed in the study to maximum.", emotion: "neutral" },
          { text: "Switch the television to the sports channel and unmute the audio.", emotion: "excited" }
        ]
      },
      {
        name: "💬 Greetings & Everyday Chat",
        sentences: [
          { text: "Good morning! How are you doing on this beautiful day?", emotion: "cheerful" },
          { text: "It is wonderful to meet you. I hope you have an incredible week.", emotion: "happy" },
          { text: "Thank you so much for your generous support, I truly appreciate it.", emotion: "happy" },
          { text: "What are your plans for the weekend? Are you going anywhere special?", emotion: "cheerful" },
          { text: "Excuse me, could you please tell me the quickest way to the train station?", emotion: "neutral" },
          { text: "Have a great evening ahead and take good care of yourself!", emotion: "calm" },
          { text: "Could you repeat that more slowly? I want to make sure I understand.", emotion: "concerned" },
          { text: "That sounds like a brilliant idea! Let us get started right away.", emotion: "excited" },
          { text: "I am feeling extremely energetic and happy today!", emotion: "excited" },
          { text: "What is your absolute favorite hobby to pass the time?", emotion: "cheerful" }
        ]
      },
      {
        name: "❓ Factual Q&A & Support",
        sentences: [
          { text: "What is the approximate distance between the Earth and the Moon?", emotion: "neutral" },
          { text: "Who was the primary architect of the Eiffel Tower in Paris?", emotion: "neutral" },
          { text: "Explain the basic concept of photosynthesis in green plants.", emotion: "neutral" },
          { text: "I would like to file a formal request to cancel my airline reservation.", emotion: "concerned" },
          { text: "Can you check my savings account balance and list the last three transactions?", emotion: "neutral" },
          { text: "What are the primary symptoms of vitamin D deficiency in young adults?", emotion: "concerned" },
          { text: "Where is the nearest 24-hour pharmacy located in this neighborhood?", emotion: "concerned" },
          { text: "I am experiencing issues logging into my secure customer portal.", emotion: "concerned" },
          { text: "When does the next scheduled flight to London Heathrow depart?", emotion: "neutral" },
          { text: "Could you summarize the main plot points of the novel Pride and Prejudice?", emotion: "neutral" }
        ]
      },
      {
        name: "💻 Tech, Code & Reasoning",
        sentences: [
          { text: "Explain the fundamental difference between a list and a tuple in Python.", emotion: "neutral" },
          { text: "How do I implement a secure token-based authentication mechanism in Node.js?", emotion: "neutral" },
          { text: "Write a recursive function to calculate the Fibonacci sequence up to N terms.", emotion: "neutral" },
          { text: "What are the main performance advantages of using a NoSQL database over SQL?", emotion: "neutral" },
          { text: "Explain the concept of time complexity and Big O notation in algorithms.", emotion: "neutral" },
          { text: "How does git rebase differ from git merge in source control workflows?", emotion: "neutral" },
          { text: "Optimize this SQL query to retrieve active users sorted by creation date.", emotion: "neutral" },
          { text: "What is the difference between synchronous and asynchronous execution in Javascript?", emotion: "neutral" },
          { text: "How does the virtual DOM improve rendering speeds in modern frontend frameworks?", emotion: "neutral" },
          { text: "Explain how a deep convolutional neural network perceives visual features.", emotion: "neutral" }
        ]
      }
    ]
  },
  {
    name: "हिन्दी (Hindi)",
    flag: "🇮🇳",
    value: "hi",
    categories: [
      {
        name: "💬 सामान्य बातचीत",
        sentences: [
          { text: "नमस्ते! आप कैसे हैं? आशा है आपका आज का दिन बहुत अच्छा रहा होगा।", emotion: "cheerful", translation: "Hello! How are you? Hope you had a very good day." },
          { text: "आपसे मिलकर मुझे बेहद खुशी हुई। आपका यहाँ बहुत-अभिवादन है।", emotion: "happy", translation: "Extremely glad to meet you. You are very welcome here." },
          { text: "आपका बहुत-बहुत धन्यवाद, आपकी सहायता मेरे लिए बहुत महत्वपूर्ण थी।", emotion: "happy", translation: "Thank you very much, your help was very important to me." },
          { text: "कृपया क्या आप मुझे बता सकते हैं कि अभी समय क्या हो रहा है?", emotion: "neutral", translation: "Could you please tell me what time it is right now?" },
          { text: "शुभरात्रि! मीठे सपने देखें और कल सुबह आराम से उठें।", emotion: "calm", translation: "Good night! Sweet dreams and wake up refreshed tomorrow." },
          { text: "आज मौसम बहुत सुहावना है, चलो बाहर टहलने चलते हैं।", emotion: "cheerful", translation: "The weather is very pleasant today, let's go for a walk outside." },
          { text: "कृपया मेरी बात ध्यान से सुनें, यह विषय बहुत ही आवश्यक है।", emotion: "neutral", translation: "Please listen to me carefully, this topic is very essential." },
          { text: "कोई बात नहीं, गलतियाँ इंसान से ही होती हैं, चिंता मत कीजिए।", emotion: "calm", translation: "No worries, humans make mistakes, please don't worry." },
          { text: "आप वीकेंड पर क्या करना पसंद करते हैं? क्या कोई विशेष योजना है?", emotion: "cheerful", translation: "What do you like to do on weekends? Any special plans?" },
          { text: "भविष्य के लिए आपको मेरी तरफ से हार्दिक शुभकामनाएँ!", emotion: "happy", translation: "Hearty wishes to you for the future!" }
        ]
      },
      {
        name: "🏠 घर और दैनिक कार्य",
        sentences: [
          { text: "रसोई घर के सभी बल्ब बंद कर दो और मुख्य दरवाज़ा लॉक कर लो।", emotion: "neutral", translation: "Turn off all kitchen bulbs and lock the main door." },
          { text: "एसी का तापमान चौबीस डिग्री पर सेट कर दो और पंखा धीमा करो।", emotion: "neutral", translation: "Set the AC temperature to 24 degrees and slow down the fan." },
          { text: "स्मार्ट टीवी चालू करो और उस पर आज के समाचार लगाओ।", emotion: "cheerful", translation: "Turn on the smart TV and play today's news on it." },
          { text: "क्या गीज़र चालू है या मैंने उसे बंद कर दिया था?", emotion: "concerned", translation: "Is the geyser on or did I turn it off?" },
          { text: "शाम को सात बजे का अलार्म लगा दो ताकि मैं समय पर उठ सकूँ।", emotion: "neutral", translation: "Set an alarm for 7 PM so that I can wake up on time." },
          { text: "म्यूजिक सिस्टम पर कोई मधुर और शांत हिन्दी गाना बजाओ।", emotion: "calm", translation: "Play some sweet and calm Hindi song on the music system." },
          { text: "बच्चों के कमरे की light को तीस प्रतिशत तक धीमा कर दो।", emotion: "calm", translation: "Dim the lights in the kids' room by 30 percent." },
          { text: "फ्रिज का दरवाज़ा खुला रह गया है, कृपया उसे बंद कर दें।", emotion: "concerned", translation: "The fridge door is left open, please close it." },
          { text: "वाशिंग मशीन में कपड़े धोने के लिए टाइमर लगा दो।", emotion: "neutral", translation: "Set a timer for washing clothes in the washing machine." },
          { text: "हॉल के परदे बंद कर दो ताकि बाहर की धूप अंदर न आए।", emotion: "neutral", translation: "Close the hall curtains so that outside sunlight doesn't come in." }
        ]
      },
      {
        name: "❓ सामान्य ज्ञान और सहायता",
        sentences: [
          { text: "भारत की राजधानी क्या है और वहाँ की मुख्य भाषाएँ कौन सी हैं?", emotion: "neutral", translation: "What is the capital of India and what are its main languages?" },
          { text: "हमारे सौरमंडल में सबसे बड़ा ग्रह कौन सा है?", emotion: "neutral", translation: "Which is the largest planet in our solar system?" },
          { text: "मुझे बैंक खाता खोलने के लिए किन-किन दस्तावेज़ों की आवश्यकता होगी?", emotion: "neutral", translation: "What documents will I need to open a bank account?" },
          { text: "दिल्ली से मुंबई जाने के लिए सबसे तेज़ ट्रेन कौन सी है?", emotion: "neutral", translation: "Which is the fastest train to go from Delhi to Mumbai?" },
          { text: "कृपया मुझे पनीर बटर मसाला बनाने की आसान विधि बताइए।", emotion: "cheerful", translation: "Please tell me an easy recipe for making Paneer Butter Masala." },
          { text: "मेरे बचत खाते में इस समय कुल कितनी धनराशि शेष है?", emotion: "neutral", translation: "What is the total balance remaining in my savings account right now?" },
          { text: "इंटरनेट कनेक्शन काम नहीं कर रहा है, मैं राउटर को कैसे रीसेट करूँ?", emotion: "concerned", translation: "Internet connection is not working, how do I reset the router?" },
          { text: "ताज़महल का निर्माण किसने और किस वर्ष में करवाया था?", emotion: "neutral", translation: "Who built the Taj Mahal and in which year?" },
          { text: "मुझे सिरदर्द की समस्या है, क्या मुझे कोई दवा लेनी चाहिए?", emotion: "concerned", translation: "I have a headache, should I take some medicine?" },
          { text: "नज़दीकी पुलिस स्टेशन का आपातकालीन फ़ोन नंबर क्या है?", emotion: "concerned", translation: "What is the emergency phone number of the nearest police station?" }
        ]
      }
    ]
  },
  {
    name: "Hinglish (Mixed)",
    flag: "🇮🇳/🇬🇧",
    value: "hinglish",
    categories: [
      {
        name: "💬 Conversational Chat",
        sentences: [
          { text: "Hi! Kaise ho yaar? Bahut dino baad baat ho rahi hai.", emotion: "cheerful", translation: "Hi! How are you friend? Talking after a long time." },
          { text: "Aaj ka kya plan hai? Kahin bahar chalein kya?", emotion: "cheerful", translation: "What's the plan for today? Shall we go out somewhere?" },
          { text: "Mujhe ek help chahiye thi, kya tum free ho abhi?", emotion: "neutral", translation: "I needed a help, are you free right now?" },
          { text: "Wo movie sach mein bahut amazing thi, tumhein bhi dekhni chahiye.", emotion: "excited", translation: "That movie was really amazing, you should watch it too." },
          { text: "Chalo, main nikalta hoon, kal subah call karta hoon.", emotion: "neutral", translation: "Okay, I am leaving, will call you tomorrow morning." },
          { text: "Actually mujhe lagta hai ki ye idea bilkul perfect hai.", emotion: "happy", translation: "Actually I think that this idea is absolutely perfect." },
          { text: "Sorry, main thoda busy tha isliye reply nahi kar paya.", emotion: "calm", translation: "Sorry, I was a bit busy so I couldn't reply." },
          { text: "Ek baar double check kar lo, kahin koi mistake na ho jaye.", emotion: "concerned", translation: "Double check once, lest any mistake happens." },
          { text: "Mera phone galti se silent pe reh gaya tha, calls miss ho gaye.", emotion: "concerned", translation: "My phone was accidentally left on silent, missed the calls." },
          { text: "Bahut badhiya kaam kiya tumne! Keep it up!", emotion: "excited", translation: "Very good job you did! Keep it up!" }
        ]
      },
      {
        name: "🏠 Tech & IoT Commands",
        sentences: [
          { text: "AC ko turn off kar do aur room ka temperature display karo.", emotion: "neutral", translation: "Turn off the AC and display the room temperature." },
          { text: "Geyser ko automatically switch on kar do please.", emotion: "neutral", translation: "Please switch on the geyser automatically." },
          { text: "Smart speaker pe thode energetic songs play karo.", emotion: "excited", translation: "Play some energetic songs on the smart speaker." },
          { text: "Living room ke fan ki speed thodi kam kar do.", emotion: "calm", translation: "Slightly reduce the speed of the living room fan." },
          { text: "Kitchen ki lights brightness level fifty percent set karo.", emotion: "neutral", translation: "Set the kitchen lights brightness level to 50 percent." },
          { text: "Microwave mein pop-corn ke liye timer activate karo.", emotion: "neutral", translation: "Activate the timer for popcorn in the microwave." },
          { text: "Kya backyard ka gate properly lock ho gaya hai?", emotion: "concerned", translation: "Is the backyard gate properly locked?" },
          { text: "TV volume down karo, abhi zoom call start hone wali hai.", emotion: "neutral", translation: "Volume down the TV, a Zoom call is about to start." },
          { text: "Subah ke saat baje ka ek fresh alarm set kar do.", emotion: "cheerful", translation: "Set a fresh alarm for 7 AM in the morning." },
          { text: "Bedroom ka thermostat seventy-five degrees Fahrenheit kar do.", emotion: "neutral", translation: "Bedroom thermostat seventy-five degrees Fahrenheit." }
        ]
      },
      {
        name: "❓ Queries & Support",
        sentences: [
          { text: "Mera internet router disconnect ho gaya hai, red light blink kar rahi hai.", emotion: "concerned", translation: "My internet router is disconnected, red light is blinking." },
          { text: "Agle week ka weather forecast check karke batao please.", emotion: "cheerful", translation: "Please check and tell the weather forecast for next week." },
          { text: "Train ticket booking status confirmed hai ya abhi bhi waiting chal rahi hai?", emotion: "concerned", translation: "Is the train ticket booking status confirmed or still waiting?" },
          { text: "Mera account balance kitna hai? Last transaction kab hui thi?", emotion: "neutral", translation: "How much is my account balance? When did the last transaction occur?" },
          { text: "Is area mein best vegetarian restaurant kaun sa hai?", emotion: "cheerful", translation: "Which is the best vegetarian restaurant in this area?" },
          { text: "Mujhe flight booking cancel karni hai, refund process kaise hoga?", emotion: "concerned", translation: "I want to cancel the flight booking, how will the refund process work?" },
          { text: "Aapka customer care number kya hai? Mujhe direct executive se baat karni hai.", emotion: "neutral", translation: "What is your customer care number? I want to speak to the executive directly." },
          { text: "Laptop lag ho raha hai, RAM clean karne ka easy way batao.", emotion: "neutral", translation: "Laptop is lagging, tell an easy way to clean RAM." },
          { text: "Mujhe prime membership cancel karne ka steps samjha do.", emotion: "neutral", translation: "Explain the steps to cancel my prime membership." },
          { text: "Nearest petrol pump ki distance kitni hai yahan se?", emotion: "neutral", translation: "What is the distance of the nearest petrol pump from here?" }
        ]
      }
    ]
  }
];
