# Automated Attributes Extraction using Large Language Models
# Flair model:
As we mentioned in our paper, two law students annotated **200 crime-related** documents. To annotate those documents they mainly used **seven** tags- _expertwittest_, _wittest_, _assault_, _riot_, _homicide_, _imprisonment_, _evidence_. <br>
To train our model, we split our dataset in the ratio of 4:1. For the training we merely used 152 documents. 
# Dataset:
For all the experiments we used **Indian Supreme Court** proceedings.
# Few-shot learning using GPT-3.5-turbo model:
For training, we used 10 sentences for _riot_ and _evidence_. <br>
In your favorite virtual environment, simply do:
```ruby
!pip install openai
```
```ruby
import openai
openai.api_key = " "
```
```ruby
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.8, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]
prompt="""
He also took note of the fact that two persons Shambhu and Avadhlal were falsely involved as accused by P.W.1.
There are many other vital discrepancies in the testimony of the eyewitnesses inasmuch as the testimony of the witnesses are at variance with the case set out in the First Information Report and as such, the High Court was justified in discarding the testimony of the witnesses.
There appears to be some inconsistency in the evidence of eyewitnesses and the medical evidence but this inconsistency is of very insignificant character
The submission was, their evidence is totally untrustworthy and suffers from material contradictions.
There had been material discrepancies/contradictions/ inconsistencies in regard to the lodging of FIR and investigation so far as the statements of Pratap Singh, Head Constable, and R.D. Yadav, S.O., and the entries made in the Rojnamcha.
The prosecution case is that A-1 and A-5 armed with lathis and spears cannot be accepted inasmuch as neither of them used any such weapon
We are therefore of the view that the credibility of his version regarding the words alleged to have been uttered by the victim is open to doubt as it goes against probabilities and the natural course of conduct
This statement can not be true because P.W.10 doctor specifically stated that the injured was not sent by the police and there was no hospital memo.
Apart from the above infirmity in the evidence of three eyewitnesses, I find that the prosecution evidence is of a partisan character and not much on which implicit reliance can be placed.
Paul's testimony thus creates some doubt regarding the reliability of the prosecution evidence that Joseph had received injury with a wooden spear at the hand of accused No. 6.
"""
custom_prompt = f"""
    Your task is to generate '10' statements that closely resemble the statements
     given in the Examples delimited by \"\"\". The statements in the example resemble a tag \'evidence_inconsistency\'.
     The generated statements should also resemble the same tag. Statements that imply some impending
     discrepancy in facts generally resemble \' evidence_inconsistency\'.
     Each statement must not have less than \"25\" words.
    Examples: \"\"\"{prompt}\"\"\"
    """
response = get_completion(custom_prompt)
print(response)
```
This should print:
```ruby
1. The witness statements seem to contradict each other on crucial points, raising concerns about their accuracy.
2. It is unclear how the prosecution can reconcile the discrepancies between the witness statements and the physical evidence.
3. The accused's alibi raises some inconsistencies, which need further investigation to establish the truth.
4. The prosecution's version of events seems to conflict with the medical evidence, casting doubt on their reliability.
5. The testimony of the key witness appears to be inconsistent with the facts of the case, leading to questions about their credibility.
6. There seem to be discrepancies between the witness statements and the official records, which need to be resolved before proceeding with the case.
7. The victim's statement contradicts that of the eyewitnesses, creating uncertainty about what really happened.
8. The accused's story appears to change every time they are questioned, making it difficult to establish the facts of the case.
9. The witness statements contain many inconsistencies, making it hard to determine what really happened.
10. The evidence presented by the prosecution doesn't seem to match up with the timeline of events, creating doubt about the veracity of their claims.
```
# Text generation using GPT-2 model:
Here we use GPT-2 pertained model. <be>
Example: <br>
```ruby
!pip install transformers
```
```ruby
import tensorflow as tf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
```
```ruby
for sent in collection:
  j=len(sent)
  encoded_input=tokenizer(i,return_tensors='pt')
  ind=encoded_input['input_ids']
  am=encoded_input['attention_mask']
  #input_ids = tokenizer.encode(i, return_tensors='pt')
  output = model.generate(ind,attention_mask=am,max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True,temperature=1.5)
  print(tokenizer.decode(output[0], skip_special_tokens=True)[j:])
```
This should print:
```ruby
The court also noted that the accused had not been produced before the court on the day of the hearing. The court further observed that it was not possible for the prosecution to prove the case beyond a reasonable doubt. It was, therefore, directed that both accused should be produced for cross-examination.

It is also important to note that there is no evidence to support the contention that the accused was in possession of a firearm at the time of his arrest. It is not possible for a person to have a gun in his possession at a time when he is being questioned by the police. There is, therefore, no basis on which to conclude that he was carrying a weapon at that time. In addition, it is clear from the evidence of other witnesses that they were not aware of any gun being

There is no evidence to support the claim that there was a deliberate attempt to cover up the truth about the events of 9/11. There is, however, some evidence that the official story is not the whole truth and that some of the facts may not be as they are being presented to the public.
```
# Judgemnet prediction:
Our primary focus in this study revolved around two types of judgments: "accept" and "reject." Consequently, this particular aspect posed a binary classification problem.
# Statute prediction:
In the context of statute prediction, our main emphasis was on five IPC sections. The descriptions of these sections are provided below: <br>
* **IPC 148**: Rioting, armed with a deadly weapon. <br>
* **IPC 300**: When culpable homicide is not murder. <br>
* **IPC 302**: Punishment for murder. <br>
* **IPC 304**: Punishment for culpable homicide not amounting to murder. <br>
* **IPC 307**: Attempt to murder.<br>

