<h1>🔧 [1D classification] Fault Diagnosis using CNN</h1>

<p>This project implements a complete modular pipeline for classifying bearing faults using <strong>Convolutional Neural Networks</strong> on the <strong>CWRU</strong> (Case Western Reserve University) dataset.</p>
# CWRU Bearing Fault Diagnosis using CNN

My name is JIA CHENGLIN
This project is a replication and extension of a baseline CNN model for CWRU bearing fault diagnosis.


Experiment 1: Random Split (Data Leakage)
Method: Randomly split the dataset into train/val/test sets (Baseline provided by the original repo).
Result: Test Accuracy 100.00%.
Observation: This is an artificially inflated result due to data leakage. Slices from the same continuous signal ended up in both training and testing sets, causing the model to "memorize" instead of "generalize".


Experiment 2: Time-Series Split (Cross-Condition Simulation)
Method: To simulate real-world cross-condition scenarios, I split the data sequentially (first 70% for training, last 20% for testing) without shuffling. This means the model was trained on "Normal" and "Inner Race" faults, but tested on "Outer Race" faults.
Result: Test Accuracy dropped to 0.00%.
Insight: This catastrophic failure reveals a critical challenge in fault diagnosis: a model trained on limited conditions cannot generalize to unseen faults. This motivated me to implement a more scientific data splitting strategy (Stratified by class) to evaluate the model's true capability.


<hr/>

<h2>📁 Project Structure</h2>

<pre>
cnn_pipeline/
├── main.py                  # Main training/evaluation script
├── arguments.py             # Command-line arguments
├── dataset.py               # DataLoader generation
├── model.py                 # CNN model definition
├── train.py                 # Training function (with early stopping)
├── test.py                  # Test function
├── preprocess.py            # CWRU .mat file preprocessing
</pre>

<hr/>

<h2>📥 Dataset</h2>

<p>
This project uses the publicly available <strong>CWRU Bearing Data</strong> from Case Western Reserve University.
You can download the dataset files from the official website below:
</p>

<p>
🔗 <a href="https://engineering.case.edu/bearingdatacenter/download-data-file" target="_blank">
https://engineering.case.edu/bearingdatacenter/download-data-file
</a>
</p>

<p>
Please place the downloaded <code>.mat</code> files inside the <code>raw_data/</code> directory:
</p>

<pre>
raw_data/
├── 97.mat     # Normal
├── 105.mat    # Inner race fault
├── 118.mat    # Ball fault
├── 130.mat    # Outer race fault
</pre>

<p>The <code>preprocess.py</code> module converts raw signals into sliding window segments for supervised classification.</p>

<hr/>

<h2>🚀 How to Run</h2>

<pre><code>pip install numpy scipy torch scikit-learn matplotlib tqdm</code></pre>

<pre><code>python main.py</code></pre>

<p>The script will:</p>
<ol>
  <li>Preprocess raw <code>.mat</code> files into (X, Y) arrays</li>
  <li>Split data into train/valid/test</li>
  <li>Train CNN using early stopping</li>
  <li>Evaluate final test accuracy</li>
</ol>

<hr/>

<h2>🧠 CNN Architecture</h2>

<p>Three-layer 1D convolutional encoder with ReLU, BatchNorm, and MaxPooling followed by fully connected classification layers.</p>

<pre>
Input → Conv1d → ReLU → MaxPool → Conv1d → ReLU → MaxPool → Conv1d → Flatten → FC layer → FC layer → Output
</pre>

<hr/>

<h2>📈 Training Options (arguments.py)</h2>

<table>
  <tr><th>Argument</th><th>Description</th><th>Default</th></tr>
  <tr><td><code>--epochs</code></td><td>Number of training epochs</td><td>10</td></tr>
  <tr><td><code>--lr</code></td><td>Learning rate</td><td>1e-4</td></tr>
  <tr><td><code>--lamda</code></td><td>LR scheduler decay</td><td>0.97</td></tr>
  <tr><td><code>--early_stop</code></td><td>Early stopping patience</td><td>20</td></tr>
  <tr><td><code>--train_size</code></td><td>Train/val split ratio</td><td>0.8</td></tr>
  <tr><td><code>--batch_size</code></td><td>Mini-batch size</td><td>64</td></tr>
</table>

<hr/>

<h2> Views </h2>

![](http://profile-counter.glitch.me/97yong-signal-fault-classification/count.svg)
