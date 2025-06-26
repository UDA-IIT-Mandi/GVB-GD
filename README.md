# GVB-GD

This is a code to implement the paper Gradually Vanishing Bridge for Adversarial Domain Adaptation,https://doi.org/10.48550/arXiv.2003.13183
code was borrowed from cuishuhao github and modified so one can easily work on it in jupyter.

I implimented this architecture for dcase dataset with passt as a feature extractor

all you need to do to test it out is download dcase dataset make the split correct (you can find the split in the dcase challenge, just google it. )


There are scripts in in the last cells of the notebook (i commented them out , to prevent them being accidently executed )below to set up text file with audio paths , just get the paths ready and you are good to go 


i have also included the modular code below to implement the paper , i suggest sticking with the jupyter version . 


<h2>Things you can try changing<h2> 

<p>Try changing the loss function of the descriminator to "Wasserstien loss function" , "domain confusion loss"<p>

<p>Try changing the generator  descriminator architecture to something else , experiment   using some transformer based architecture for the generator<p> 

<p>also try changing the feature extractor model<p>

;)
