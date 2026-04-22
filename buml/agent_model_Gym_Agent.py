###############
# AGENT MODEL #
###############
import datetime
from besser.BUML.metamodel.state_machine.state_machine import Body, Condition, ConfigProperty, CustomCodeAction
from besser.BUML.metamodel.state_machine.agent import Agent, AgentReply, LLMReply, RAGReply, DBReply, LLMOpenAI, LLMHuggingFace, LLMHuggingFaceAPI, LLMReplicate, RAGVectorStore, RAGTextSplitter
from besser.BUML.metamodel.structural import Metadata
import operator

agent = Agent('Gym_Agent')

agent.add_property(ConfigProperty('websocket_platform', 'websocket.host', '0.0.0.0'))
agent.add_property(ConfigProperty('websocket_platform', 'websocket.port', 8765))
agent.add_property(ConfigProperty('websocket_platform', 'streamlit.host', '0.0.0.0'))
agent.add_property(ConfigProperty('websocket_platform', 'streamlit.port', 5000))
agent.add_property(ConfigProperty('nlp', 'nlp.language', 'en'))
agent.add_property(ConfigProperty('nlp', 'nlp.region', 'US'))
agent.add_property(ConfigProperty('nlp', 'nlp.timezone', 'Europe/Madrid'))
agent.add_property(ConfigProperty('nlp', 'nlp.pre_processing', True))
agent.add_property(ConfigProperty('nlp', 'nlp.intent_threshold', 0.4))
agent.add_property(ConfigProperty('nlp', 'nlp.openai.api_key', 'YOUR-API-KEY'))
agent.add_property(ConfigProperty('nlp', 'nlp.hf.api_key', 'YOUR-API-KEY'))
agent.add_property(ConfigProperty('nlp', 'nlp.replicate.api_key', 'YOUR-API-KEY'))

# INTENTS
Muscles_intent = agent.new_intent('Muscles_intent', [
    'I want muscles',
],
description="Question about how to become muscular and which exercises to perform.")
Nutrition_intent = agent.new_intent('Nutrition_intent', [
    'Which food to eat?',
],
description="Question about nutrition in gym.")
Other = agent.new_intent('Other', [
],
description="Any question that does not fit in \"Muscles\" or \"Nutrition\"")

# Create LLM instance for use in state bodies
llm = LLMOpenAI(agent=agent, name='gpt-4o-mini', parameters={})

# STATES
Initial = agent.new_state('Initial', initial=True)
Idle = agent.new_state('Idle')
TrainingPlan = agent.new_state('TrainingPlan')
Nutrition = agent.new_state('Nutrition')
OtherQuestions = agent.new_state('OtherQuestions')

# Initial state
Initial_body = Body('Initial_body')
Initial_body.add_action(AgentReply('Hi, I am your buddy, the fitness agent.'))

Initial.set_body(Initial_body)
Initial.go_to(Idle)

# Idle state
Idle_body = Body('Idle_body')
Idle_body.add_action(AgentReply('I am here to answer any questions regarding exercises, nutrition and recovery.'))

Idle.set_body(Idle_body)
Idle_fallback_body = Body('Idle_fallback_body')
Idle_fallback_body.add_action(AgentReply('I focus on gym and fitness topics only dude.'))

Idle.set_fallback_body(Idle_fallback_body)
Idle.when_intent_matched(Muscles_intent).go_to(TrainingPlan)

Idle.when_intent_matched(Other).go_to(OtherQuestions)

Idle.when_intent_matched(Nutrition_intent).go_to(Nutrition)

# TrainingPlan state
TrainingPlan_body = Body('TrainingPlan_body')
TrainingPlan_body.add_action(AgentReply('Focus on heavy compound lifts like squats, deadlifts, bench press, overhead press, and rows to build overall muscle mass. '))
TrainingPlan_body.add_action(AgentReply('Train 3–5 times per week, progressively increasing the weight while eating enough protein and calories to support growth. '))
TrainingPlan_body.add_action(AgentReply('Get good sleep and stay consistent—muscle comes from steady effort over time.'))

TrainingPlan.set_body(TrainingPlan_body)
TrainingPlan.go_to(Idle)

# Nutrition state
Nutrition_body = Body('Nutrition_body')
Nutrition_body.add_action(AgentReply('Nutrition basics: Eat mostly whole foods (lean protein, vegetables, fruits, whole grains, healthy fats). Protein: ~1.6–2.2 g per kg of body weight daily..'))
Nutrition_body.add_action(AgentReply('Match calories to your goal: slight surplus to gain muscle, deficit to lose fat, maintenance to stay the same. '))
Nutrition_body.add_action(AgentReply(' Carbs fuel training; fats support hormones.'))
Nutrition_body.add_action(AgentReply(' Drink plenty of water and limit ultra-processed foods and alcohol. '))
Nutrition_body.add_action(AgentReply('Be consistent — results come from habits, not perfection'))

Nutrition.set_body(Nutrition_body)
Nutrition.go_to(Idle)

# OtherQuestions state
OtherQuestions_body = Body('OtherQuestions_body')
OtherQuestions_body.add_action(LLMReply())

OtherQuestions.set_body(OtherQuestions_body)
OtherQuestions.go_to(Idle)

