<h1>Run Experiments</h1>
All the execution lines for running the bash scripts can be found under the <mark>run_yelp</mark> folder, which calls the corresponding bash script files specified.

The bash scripts in this folder are used to generate results for the 3 experiments presented in the paper. 
Comments in the scripts indicates the corresponding experiments (i.e., Experiment 1., Experiment 2., Experiment 3.)

<h1>Bayesian Update Hyperparameters</h1>
Since this paper leverages the Bayesian update methodology used by [BK-VAE](https://ssanner.github.io/papers/sigir21_tcavcrit.pdf), 
we also inherit the set up for using two hyperparameters for <b>precision</b> and <b>noise</b>. The following table lists out the hyperparameters used by the 3 experiments presentend in the paper.
 
<table>
  <tr>
    <td></td>
    <td></td>
    <td colspan="2" align="center"><a href="https://drive.google.com/file/d/1zDLsmd69hb6FDoqS_SbqYVrgAPaVe6K0/view?usp=sharing">Yelp</a></td>
    <td colspan="2" align="center"><a href="https://drive.google.com/file/d/1d3CiT_wDspVf87my4eo6plpzyTa3huuz/view?usp=sharing">ML10M</a></td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td align="center">Precision</td>
    <td align="center">Noise</td>
    <td align="center">Precision</td>
    <td align="center">Noise</td>
  </tr>
  <tr>
    <td rowspan="2" align="center">Experiment 1. <br> Pure Critiquing task</td>
    <td align="center">{BK / DCE}-EXP</td>
    <td rowspan="2" align="center">10</td>
    <td rowspan="2" align="center">0</td>
    <td rowspan="2" align="center">1</td>
    <td rowspan="2" align="center">10</td>
  </tr>
  <tr>
    <td align="center">{BK / DCE}-Normal</td>
  </tr>
  <tr>
    <td rowspan="3" align="center">Experiment 2. <br> Clarification Critiquing task</td>
    <td align="center">BK-NN</td>
    <td rowspan="3" align="center">100</td>
    <td rowspan="3" align="center">0</td>
    <td rowspan="3" align="center">1</td>
    <td rowspan="3" align="center">3</td>
  </tr>
  <tr>
    <td align="center">BK-random</td>
  </tr>
  <tr>
    <td align="center">DCE-Tree</td>
  </tr>
  <tr>
    <td rowspan="3" align="center">Experiment 3. <br> Clarification Performance</td>
    <td align="center">non-personalized clarification</td>
    <td rowspan="3" align="center">100</td>
    <td rowspan="3" align="center">0</td>
    <td rowspan="3" align="center">1</td>
    <td rowspan="3" align="center">0</td>
  </tr>
  <tr>
    <td align="center">personalized-clarification</td>
  </tr>
  <tr>
    <td align="center">without-clarification</td>
  </tr>
</table>
