# Automated Attributes Extraction using Large Language Models
# Flair model:
As we mentioned in our paper, two law students annotated **200 crime-related** documents. To annotate those documents they mainly used **seven** tags- _expertwittest_, _wittest_, _assault_, _riot_, _homicide_, _imprisonment_, _evidence_. <br>
To train our model, we split our dataset in the ratio of 4:1. For the training we merely used 152 documents. 
# Dataset:
For all the experiments we used **Indian Supreme Court** proceedings.
# Few-shot learning using GPT-3.5-turbo model:

# Judgemnet prediction:
Our primary focus in this study revolved around two types of judgments: "accept" and "reject." Consequently, this particular aspect posed a binary classification problem.
# Statute prediction:
In the context of statute prediction, our main emphasis was on five IPC sections. The descriptions of these sections are provided below: <br>
**IPC 148**: Rioting, armed with a deadly weapon. <br>
**IPC 300**: When culpable homicide is not murder. <br>
**IPC 302**: Punishment for murder. <br>
**IPC 304**: Punishment for culpable homicide not amounting to murder. <br>
**IPC 307**: Attempt to murder.<br>

