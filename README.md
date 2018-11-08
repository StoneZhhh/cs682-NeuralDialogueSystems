# cs682-NeuralDialogueSystems

## Description
### Original idea
The idea is first published in the paper https://arxiv.org/pdf/1703.01008.pdf End-to-End Task-Completion Neural Dialogue Systems. The team implemented their version in https://github.com/MiuLab/TC-Bot.

### What we are planning to do
* Re-implement the system
* Add some new features, typically some reinforcement learning tricks in some modules like System Action/Policy.

## Milestone
### Some basic concepts
* **IOB format**, short for inside, outside, beginning. 
    * 'B-' prefix before a tag means the tag is beginning of a chunk.
    * 'I-' prefix before a tag means the tag is in a chunk.
    * 'O' tag indicates that a token belongs to no chunk (has no tag).

* **slot filling**, by the paper *Investigation of Recurrent-Neural-Network Architectures and Learning Methods for Spoken Language Understanding*,
   
  >For the slot filling task, the input is the sentence consisting of a sequence of words, and the output is a sequence of slot/concept IDs, one for each word. 
 
* **intent prediction**, we input a word sequence $\vec{x}$ and get the predicted slots though $\vec{y} = LSTM(\vec{s})$. Still not clear how to get intention using the predicted slots.  


### Create the project
* Decide the first version of project structure (psv1).

* Decide the first version of algorithms and models for each module .
    * Language Understanding (LU) module can be simply implemented using a single LSTM. Its task is intent prediction and slot filling. The math expression is that given $\vec{x} = (w_1, w_2,  \cdots, w_n, EOS)$ is a sequence of words, $\vec{y} = (s_1, s_2, \cdots, c_n)$ is the associated slots, the LU model has to implement the affine from $\vec{x}$ to $\vec{y}$, and the solution is $\vec{y} = LSTM(\vec{s})$. This process is kind of like what we have done in cs682 assignment 3. The weight of the LSTM model is trained based on maximizing the conditional probability of the slots and the intent $\vec{y}$ given the word sequence $\vec{x}$. We expect to get IOB format slots from $\vec{x}$ using LSTM, and we need to determine the intention using the IOB format slots.
    * Additional explanation for LU using an example:
         * Words as input $\vec{x}$: (find, action, movies, this, weekend).
         * After deploying LSTM we get IOB format Slots: (O, B-genre, O, B-date, I-date).
         * After some unknown magic we get predicted intention: (find_movie)
     
    * Dialogue Management (DM) module processes the IOB format slots and predicted intention got in LU module. DM module includes two stages.
         * Given slots and their corresponding values, we cast some unknown magic to get a state (of our dialogue system)
         * Do policy improvement using current system state mentioned above, user actions based on slots (e.g., `request(moviename; genre=action; date=this weekend))`)  and system action based on intention (e.g., `request(location)`, which is equal to find_movie given movie name and time). Have no clear idea now how to do this policy improvement with two actions. Only have experience with (s, a) style reinforcement learning. Since P and R are unknown, TD or Monte Carlo approach seem to work. Another thing for policy improvement, We have a predefined policy in `dia_act_nl_pairs.v6.json`, which may be helpful as an initial policy.
    * User agent model processes a Markov-like decision process. In each time $t$, the model has a state $s_t$, containing the agent action $Am_t$, constraints $C_t$ and request $R_t$, that is  to say $s_t = (Am_t, C_t, R_t)$. It updates the state to $s_{t+1}$ using $s_t$ and the user action at time $t$ ($au_{t}$). The difference between our user agent model and traditional MDP is that it contains multi-action roles, we try to modify this to a traditional one by adding the user agent action into the environment.
    * Natural Language Generation (NLG) module generates natural language texts given the users' action. We first limit our NLG to template-based NLG only. The generating process is that the module takes the users's action as the input and generates test structure like 'I think you can see \<movie-name\> at \<cinema-name\>', using slots to represent concrete information need to be filled in. This can be done by using LSTM decoder. Then a post-processing scan is performed to replace the slot placeholders with their actual values.
  
  
* Fetch all the required data and redesign the data structure if needed.
    * `dia_acts.txt` describes the types of actions our dialogue system can respond, including request, inform, confirm_question and so on _Called in User Agent module and NLG module_.
    * `slot_set.txt` describes all the supported slots. _Called in LU module_.
    * `dia_act_nl_pairs.v6.json` describes some predefined rules about how NLG model generate dialogues. Can we get rid of this stupid limits and create an engine that generate dialogue with only some hyper parameters? _Called in DM module and NLG module_.

* Start coding!



