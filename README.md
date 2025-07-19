# Complex Word Identification with PyTorch 🔥
Sequence Labeling BiLSTM-CRF Model for the CWI Shared Task. Implemented based on the paper by <a href="https://aclanthology.org/P19-1109.pdf">Gooding and Kochmar (2019)</a> and the <a href="https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch">Github repository</a> by Zubin Gou.

# Performance 
Achieved a Macro-averaged F1 Score of 0.8691 in the combined words-only dataset.

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
    <td>7187</td>
    <td>6799</td>
    <td>388</td>
    <td>94.601</td>
  </tr>
  <tr>
    <td>C</td>
    <td>1301</td>
    <td>219</td>
    <td>1082</td>
    <td>83.167</td>
  </tr>
</table>
