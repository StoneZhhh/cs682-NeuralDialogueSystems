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
    * Language Understanding (LU) module can be simply implemented using a single LSTM. Its task is intent prediction and slot filling. The math expression is that given $\vec{x} = (w_1, w_2,  \cdots, w_n, EOS)$ is a sequence of words, $\vec{y} = (s_1, s_2, \cdots, c_n)$ is the associated slots, the LU model has to implement the affine from $\vec{x}$ to $\vec{y}$, and the solution is $\vec{y} = LSTM(\vec{s})$. This process is kind of like what we have done in cs682 assignment 3. The weight of the LSTM model is trained based on maximize the conditional probability of the slots and the intent $\vec{y}$ given the word sequence $\vec{x}$. We expect to get IOB format slots from $\vec{x}$ using LSTM, and we need to determine the intention using the IOB format slots.
    * Additional explanation for LU using an example:
         * Words as input $\vec{x}$: (find, action, movies, this, weekend).
         * After deploying LSTM we get IOB format Slots: (O, B-genre, O, B-date, I-date).
         * After some unknown magic we get predicted intention: (find_movie)
     
    * Dialogue Management (DM) module
  
* Fetch all the required data and redesign the data structure if needed.
    * `dia_acts.txt` describes the types of actions our dialogue system can respond, including request, inform, confirm_question and so on.
    * `slot_set.txt` describes all the supported slots. _Called in LU module_.
    * `dia_act_nl_pairs.v6.json` describes some predefined rules about how NLG model generate dialogues. Can we get rid of this stupid limits and create an engine that generate dialogue with only some hyper parameters?



