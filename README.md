# LDA
LDA experiment on Thai financial data

## Load final data (optional)
Run the script to get the dataset for training reproduce.

```bash
gdown.sh
```


## LDA default
![image info](./coherence_scores_default.png)
Using 50 topics as a training parameters. Train script can be reproduce at ```script/run.py```

LDA visualization at: ```lda_visualization_default.html```
Topic extraction at: ```snapshot_results_default.ipynb```


## LDA Tf-idf
![image info](./coherence_scores_tfidf.png)
Using 35 topics as a training parameters. Train script can be reproduce at ```script/run_tfidf.py```

LDA visualization at: ```lda_visualization_tfidf.html```
Topic extraction at: ```snapshot_results_tfidf.ipynb```