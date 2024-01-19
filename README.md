# WordFraud
A simple script that guides you to victory at Wordfeud. 

Built on various CNNs to transcribe the board and greedily search for the
highest scoring word in the next round. 

Currently only supports Swedish and iPhone 13/14. The solver script is however easily modifiable for any language and any device, as all you have to do is change the dictionary (as of writing this is SAOL_AUGMENTED.txt) and scale the input image accordingly. 

# Usage
Install the required modules by running the following command:

pip install -r requirements.txt

Then run the solver script using a screenshot YourScreenshot of the board:

python solver.py YourScreenshot

Optionally you can provide a parameter --N to include the top N highest scoring words. This is useful since sometimes the highest scoring word is not in Wordfeud's dictionary. 

# Example
![Highest scoring word is VALKNUT](https://github.com/yaddayaddayaddayadda/WordFraud/blob/main/125_0.png?raw=true)

