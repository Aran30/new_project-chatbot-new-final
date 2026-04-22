# You may need to add your working directory to the Python path. To do so, uncomment the following lines of code
# import sys
# sys.path.append("/Path/to/directory/besser-agentic-framework") # Replace with your directory path

import json
import logging
import operator

from baf.core.agent import Agent
from baf.library.transition.events.base_events import *
from baf.nlp.llm.llm_huggingface import LLMHuggingFace
from baf.nlp.llm.llm_huggingface_api import LLMHuggingFaceAPI
from baf.nlp.llm.llm_openai_api import LLMOpenAI
from baf.nlp.llm.llm_replicate_api import LLMReplicate
from baf.core.session import Session
from baf.nlp.intent_classifier.intent_classifier_configuration import LLMIntentClassifierConfiguration, SimpleIntentClassifierConfiguration
from baf.nlp.speech2text.openai_speech2text import OpenAISpeech2Text
from baf.nlp.text2speech.openai_text2speech import OpenAIText2Speech

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime}: {message}', style='{')


# Create the bot
agent = Agent('Gym_Agent')
# Load bot properties stored in a dedicated file
agent.load_properties('config.yaml')

# Define the platform your chatbot will use





platform = agent.use_websocket_platform(use_ui=True)
# LLM instantiation based on config['llm']
reply_llm = LLMOpenAI(
    agent=agent,
    name='gpt-5',
    parameters={}
)





ic_config = LLMIntentClassifierConfiguration(
    llm_name='gpt-5',
    parameters={},
    use_intent_descriptions=True,
    use_training_sentences=True,
    use_entity_descriptions=False,
    use_entity_synonyms=False
)

agent.set_default_ic_config(ic_config)



##############################
# INTENTS
##############################
Muscles_intent = agent.new_intent('Muscles_intent', [
    'I want muscles',
    ],
    description='Question about how to become muscular and which exercises to perform.'
)
Nutrition_intent = agent.new_intent('Nutrition_intent', [
    'Which food to eat?',
    ],
    description='Question about nutrition in gym.'
)
Other = agent.new_intent('Other', [
    ],
    description='Any question that does not fit in "Muscles" or "Nutrition"'
)



##############################
# CUSTOM CONDITIONS
##############################


##############################
# STATES
##############################


Initial = agent.new_state('Initial', initial=True)
Idle = agent.new_state('Idle')
TrainingPlan = agent.new_state('TrainingPlan')
Nutrition = agent.new_state('Nutrition')
OtherQuestions = agent.new_state('OtherQuestions')



# Initial
def Initial_body(session: Session):
    reply_text = 'Hi, I am your buddy, the fitness agent.'
    session.reply(reply_text)
Initial.set_body(Initial_body)
Initial.go_to(Idle)
# Idle
def Idle_body(session: Session):
    reply_text = 'I am here to answer any questions regarding exercises, nutrition and recovery.'
    session.reply(reply_text)
Idle.set_body(Idle_body)
def Idle_fallback_body(session: Session):
    reply_text = 'I focus on gym and fitness topics only dude.'
    session.reply(reply_text)
Idle.set_fallback_body(Idle_fallback_body)
Idle.when_intent_matched(Muscles_intent).go_to(TrainingPlan)
Idle.when_intent_matched(Other).go_to(OtherQuestions)
Idle.when_intent_matched(Nutrition_intent).go_to(Nutrition)
# TrainingPlan
def TrainingPlan_body(session: Session):
    reply_text = 'Focus on heavy compound lifts like squats, deadlifts, bench press, overhead press, and rows to build overall muscle mass. '
    session.reply(reply_text)
    reply_text = 'Train 3–5 times per week, progressively increasing the weight while eating enough protein and calories to support growth. '
    session.reply(reply_text)
    reply_text = 'Get good sleep and stay consistent—muscle comes from steady effort over time.'
    session.reply(reply_text)
TrainingPlan.set_body(TrainingPlan_body)
TrainingPlan.go_to(Idle)
# Nutrition
def Nutrition_body(session: Session):
    reply_text = 'Nutrition basics: Eat mostly whole foods (lean protein, vegetables, fruits, whole grains, healthy fats). Protein: ~1.6–2.2 g per kg of body weight daily..'
    session.reply(reply_text)
    reply_text = 'Match calories to your goal: slight surplus to gain muscle, deficit to lose fat, maintenance to stay the same. '
    session.reply(reply_text)
    reply_text = ' Carbs fuel training; fats support hormones.'
    session.reply(reply_text)
    reply_text = ' Drink plenty of water and limit ultra-processed foods and alcohol. '
    session.reply(reply_text)
    reply_text = 'Be consistent — results come from habits, not perfection'
    session.reply(reply_text)
Nutrition.set_body(Nutrition_body)
Nutrition.go_to(Idle)
# OtherQuestions
def OtherQuestions_body(session: Session):
    message = reply_llm.predict(message=session.event.message, session=session)
    session.reply(message)
OtherQuestions.set_body(OtherQuestions_body)
OtherQuestions.go_to(Idle)




# RUN APPLICATION

if __name__ == '__main__':
    agent.run()