Download the IIIT\_STR and Sports10K datasets and uncompress them into this folder.

[The IIIT Scene Text Retrieval (STR) Dataset](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-scene-text-retrieval-str-dataset)

[Sports-10K and TV Series-1M Video Datasets](https://cvit.iiit.ac.in/research/projects/cvit-projects/sports-10k-and-tv-series-1m-video-datasets)

In the case of the IIIT\_STR dataset we have found that the zip file contains several non-JPEG images with a "jpg" extension. If not fixed this causes an error in our code when trying to load them. The following command lines can be used to fix this issue:

```
unzip IIIT_STR_V1.0.zip

cd IIIT_STR_V1.0/ 

for i in `ls imgDatabase/img_*`; do for j in `identify $i | grep -v JPEG | cut -d " " -f 1`; do convert $j tmp.jpg; mv tmp.jpg $j; done; done
```
