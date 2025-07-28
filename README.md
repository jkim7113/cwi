# Complex Word Identification with PyTorch 🔥
Sequence Labeling BiLSTM-CRF Model for the CWI Shared Task. Implemented based on the paper by <a href="https://aclanthology.org/P19-1109.pdf">Gooding and Kochmar (2019)</a> and the <a href="https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch">Github repository</a> by Zubin Gou.

# Performance 
Achieved a Macro-averaged F1 Score of 0.8583 in the combined words-only dataset.

Confusion Matrix
<table>
  <tr>
    <th>NE</th>
    <th>Total</th>
    <th>N</th>
    <th>C</th>
    <th>Percent</th>
  </tr>
  <tr>
    <td>N</td>
    <td>7187.0</td>
    <td>6765.0</td>
    <td>422.0</td>
    <td>94.128</td>
  </tr>
  <tr>
    <td>C</td>
    <td>1301.0</td>
    <td>238.0</td>
    <td>1063.0</td>
    <td>81.706</td>
  </tr>
</table>

# Model Checkpoint
The checkpoint of this model is available <a href="https://drive.google.com/drive/folders/1m9yWT1sIoR42j32i_51dpTDh5gPP0hS9?usp=sharing">here</a>.
