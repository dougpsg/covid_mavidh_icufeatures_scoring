# COVID ICU features and severity scoring

This repo comprises the code for two papers [1, 2] by the Machine Vision ad Digital Health (MaViDH) research group, a collaboration between Charles Sturt University (Australia) and University of Technology Sydney.

Since large data sets are required, the notebooks were made to allow two types of execution, one of which is simpler, using pre-calculated features. Moreover, as some images have been selected and gone to slight manual treatment, we have also the name of the files available for replication. The main notebooks for the features have the prefix "testbench" because that are many parameters and ways to set up the learning of models; they were then made to facilitate experimentation and tunning.

For the simpler route, only the "testbench_classificatin_analysis" notebook is needed. It calls the pre-calculated files in the output directory and performs the learning and analyses.
To run the whole process from raw datasets to image processing and feature extraction, one will need to start on the "testbench_processing" notebook and have the data sets, and libraries commented below. 

## Libraries and data sets required

This code is based on other remarkable work and public data sets in the literature. 
If the one wants to run the process from scratch, a "datasets" directory needs to be in the root folder, which should contain the "covid-chestxray-dataset" repo from Cohen [3], the BIM-COVID19 [4], and the "Hanno" data set [6]. 
This is the reason why pre-processed images for the Cohen and Hanno data set are included in the output folder, enabling the checking of the classification analyses without needing all the image pre-processing (which is not the case for the BIM-CV data set given its size).
Moreover, to use the pre-trained specialized network from Cohen, the repo "torchxrayvision" [5] also needs to be in the root folder. 

Since the pre-processed images also have some manual adjustments like the cropping of areas that were not properly segmented, results might differ slightly if the process is run from scratch, but main results should hold. 

# References 

## Our papers

[1] Gomes, D.P., Ulhaq, A., Paul, M., Horry, M.J., Chakraborty, S., Saha, M., Debnath, T. and Rahaman, D.M., 2020. Potential features of icu admission in x-ray images of covid-19 patients. arXiv preprint arXiv:2009.12597.
<pre><code>
@misc{gomes2020potential,
      title={Potential Features of ICU Admission in X-ray Images of COVID-19 Patients}, 
      author={Douglas P. S. Gomes and Anwaar Ulhaq and Manoranjan Paul and Michael J. Horry and Subrata Chakraborty and Manas Saha and Tanmoy Debnath and D. M. Motiur Rahaman},
      year={2020},
      eprint={2009.12597},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
</pre></code>

[2] Gomes, D.P., Horry, M.J., Ulhaq, A., Paul, M., Chakraborty, S., Saha, M., Debnath, T. and Rahaman, D.M., 2020. MAVIDH Score: A COVID-19 Severity Scoring using Interpretable Chest X-Ray Pathology Features. arXiv preprint arXiv:2011.14983.
<pre><code>
@misc{gomes2020mavidh,
      title={MAVIDH Score: A COVID-19 Severity Scoring using Chest X-Ray Pathology Features}, 
      author={Douglas P. S. Gomes and Michael J. Horry and Anwaar Ulhaq and Manoranjan Paul and Subrata Chakraborty and Manash Saha and Tanmoy Debnath and D. M. Motiur Rahaman},
      year={2020},
      eprint={2011.14983},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
</pre></code>

## Required data sets and repos

[3] Cohen, J.P., Morrison, P., Dao, L., Roth, K., Duong, T.Q. and Ghassemi, M., 2020. Covid-19 image data collection: Prospective predictions are the future. arXiv preprint arXiv:2006.11988. - [Repo](https://github.com/ieee8023/covid-chestxray-dataset)

[4] Vayá, M.D.L.I., Saborit, J.M., Montell, J.A., Pertusa, A., Bustos, A., Cazorla, M., Galant, J., Barber, X., Orozco-Beltrán, D., Garcia, F. and Caparrós, M., 2020. BIMCV COVID-19+: a large annotated dataset of RX and CT images from COVID-19 patients. arXiv preprint arXiv:2006.01174. - [Repo](https://github.com/BIMCV-CSUSP/BIMCV-COVID-19)

[5] Cohen, J.P., Viviano, J., Hashir, M. and Bertrand, H., 2020. TorchXRayVision: A library of chest X-ray datasets and models. 2020.- [Repo](https://github.com/mlmed/torchxrayvision)

[6] COVID-19 Image Repository. Institute for Diagnostic and Interventional Radiology, Hannover Medical School, Hannover, Germany. - [Repo](https://github.com/ml-workgroup/covid-19-image-repository)
