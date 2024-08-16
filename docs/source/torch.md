Welcome to the pyTorch users page. The instructions below outline how to compute various UQ quantities like aleatoric and epistemic using different modeling approaches.

Overall, there is (1) one script to train regression models and (2) one to train categorical models. Let us review the configuration file first, then we will train models. 

(1) Currently, for regression problems only the Amini-evidential MLP and a standard multi-task MLP (e.g. one that does not predict uncertaintes). Support for the Gaussian model will be added eventually. To train a regression MLP