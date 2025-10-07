# Alzheimer’s Disease Classification from EEG using a Multiscale Temporal Deep Network

Original code of the paper "Alzheimer’s Disease Classification from EEG using a Multiscale Temporal Deep Network"

## Subjects Split

For each dataset here are listed the subjects from training validation and test splits. Those splits are inherited from the original work from Wang et al. (see the paper for further details), with the exception of BrainLat dataset.

| ADSZ   |            |      | APAVA  |            |      | ADFTD-Binary |            |      | ADFTD  |            |      | BrainLat           |                    |                    |
|--------|------------|------|--------|------------|------|--------------|------------|------|--------|------------|------|--------------------|--------------------|--------------------|
| Train  | Validation | Test | Train  | Validation | Test | Train        | Validation | Test | Train  | Validation | Test | Train              | Validation         | Test               |
| 25     | 39         | 44   | 3      | 15         | 1    | 37           | 54         | 60   | 37     | 54         | 60   | 1_AD_AR_sub-30020  | 1_AD_AR_sub-30009  | 1_AD_CL_sub-30005  |
| 26     | 40         | 45   | 4      | 16         | 2    | 38           | 55         | 61   | 38     | 55         | 61   | 1_AD_CL_sub-30007  | 1_AD_CL_sub-30028  | 1_AD_CL_sub-30030  |
| 27     | 41         | 46   | 5      | 19         | 17   | 39           | 56         | 62   | 39     | 56         | 62   | 1_AD_AR_sub-30008  | 1_AD_CL_sub-30033  | 1_AD_AR_sub-30004  |
| 28     | 42         | 47   | 6      | 20         | 18   | 40           | 57         | 63   | 40     | 57         | 63   | 1_AD_AR_sub-30001  | 1_AD_CL_sub-30016  | 1_AD_CL_sub-30035  |
| 29     | 43         | 48   | 7      |            |      | 41           | 58         | 64   | 41     | 58         | 64   | 1_AD_AR_sub-30015  | 1_AD_AR_sub-30018  | 1_AD_CL_sub-30019  |
| 30     | 15         | 20   | 8      |            |      | 42           | 59         | 65   | 42     | 59         | 65   | 1_AD_CL_sub-30025  | 1_AD_CL_sub-30024  | 1_AD_AR_sub-30017  |
| 31     | 16         | 21   | 9      |            |      | 43           | 22         | 29   | 43     | 79         | 84   | 1_AD_AR_sub-30002  | 1_AD_AR_sub-30011  | 1_AD_CL_sub-30027  |
| 32     | 17         | 22   | 10     |            |      | 44           | 23         | 30   | 44     | 80         | 85   | 1_AD_CL_sub-30034  | 5_HC_CL_sub-100014 | 5_HC_CL_sub-10005  |
| 33     | 18         | 23   | 11     |            |      | 45           | 24         | 31   | 45     | 81         | 86   | 1_AD_CL_sub-30010  | 5_HC_AR_sub-100024 | 5_HC_AR_sub-100018 |
| 34     | 19         | 24   | 12     |            |      | 46           | 25         | 32   | 46     | 82         | 87   | 1_AD_AR_sub-30013  | 5_HC_AR_sub-100020 | 5_HC_CL_sub-100011 |
| 35     |            |      | 13     |            |      | 47           | 26         | 33   | 47     | 83         | 88   | 1_AD_AR_sub-30012  | 5_HC_AR_sub-10006  | 5_HC_AR_sub-10002  |
| 36     |            |      | 14     |            |      | 48           | 27         | 34   | 48     | 22         | 29   | 1_AD_AR_sub-30031  | 5_HC_CL_sub-100037 | 5_HC_CL_sub-100029 |
| 37     |            |      | 21     |            |      | 49           | 28         | 35   | 49     | 23         | 30   | 1_AD_AR_sub-30026  | 5_HC_CL_sub-100034 | 5_HC_AR_sub-100031 |
| 38     |            |      | 22     |            |      | 50           |            | 36   | 50     | 24         | 31   | 1_AD_AR_sub-30029  |                    |                    |
| 1      |            |      | 23     |            |      | 51           |            |      | 51     | 25         | 32   | 1_AD_AR_sub-30022  |                    |                    |
| 2      |            |      |        |            |      | 52           |            |      | 52     | 26         | 33   | 1_AD_CL_sub-30003  |                    |                    |
| 3      |            |      |        |            |      | 53           |            |      | 53     | 27         | 34   | 1_AD_CL_sub-30014  |                    |                    |
| 4      |            |      |        |            |      | 1            |            |      | 66     | 28         | 35   | 1_AD_CL_sub-30023  |                    |                    |
| 5      |            |      |        |            |      | 2            |            |      | 67     |            | 36   | 1_AD_CL_sub-30032  |                    |                    |
| 6      |            |      |        |            |      | 3            |            |      | 68     |            |      | 1_AD_CL_sub-30006  |                    |                    |
| 7      |            |      |        |            |      | 4            |            |      | 69     |            |      | 1_AD_CL_sub-30021  |                    |                    |
| 8      |            |      |        |            |      | 5            |            |      | 70     |            |      | 5_HC_AR_sub-10004  |                    |                    |
| 9      |            |      |        |            |      | 6            |            |      | 71     |            |      | 5_HC_AR_sub-10009  |                    |                    |
| 10     |            |      |        |            |      | 7            |            |      | 72     |            |      | 5_HC_AR_sub-100035 |                    |                    |
| 11     |            |      |        |            |      | 8            |            |      | 73     |            |      | 5_HC_AR_sub-100015 |                    |                    |
| 12     |            |      |        |            |      | 9            |            |      | 74     |            |      | 5_HC_CL_sub-100016 |                    |                    |
| 13     |            |      |        |            |      | 10           |            |      | 75     |            |      | 5_HC_AR_sub-100028 |                    |                    |
| 14     |            |      |        |            |      | 11           |            |      | 76     |            |      | 5_HC_AR_sub-10003  |                    |                    |
|        |            |      |        |            |      | 12           |            |      | 77     |            |      | 5_HC_CL_sub-100017 |                    |                    |
|        |            |      |        |            |      | 13           |            |      | 78     |            |      | 5_HC_AR_sub-100022 |                    |                    |
|        |            |      |        |            |      | 14           |            |      | 1      |            |      | 5_HC_AR_sub-10007  |                    |                    |
|        |            |      |        |            |      | 15           |            |      | 2      |            |      | 5_HC_AR_sub-100033 |                    |                    |
|        |            |      |        |            |      | 16           |            |      | 3      |            |      | 5_HC_CL_sub-100010 |                    |                    |
|        |            |      |        |            |      | 17           |            |      | 4      |            |      | 5_HC_AR_sub-100038 |                    |                    |
|        |            |      |        |            |      | 18           |            |      | 5      |            |      | 5_HC_AR_sub-100026 |                    |                    |
|        |            |      |        |            |      | 19           |            |      | 6      |            |      | 5_HC_CL_sub-100021 |                    |                    |
|        |            |      |        |            |      | 20           |            |      | 7      |            |      | 5_HC_CL_sub-10001  |                    |                    |
|        |            |      |        |            |      | 21           |            |      | 8      |            |      | 5_HC_CL_sub-10008  |                    |                    |
|        |            |      |        |            |      |              |            |      | 9      |            |      | 5_HC_CL_sub-100043 |                    |                    |
|        |            |      |        |            |      |              |            |      | 10     |            |      | 5_HC_AR_sub-100012 |                    |                    |
|        |            |      |        |            |      |              |            |      | 11     |            |      | 5_HC_AR_sub-100030 |                    |                    |
|        |            |      |        |            |      |              |            |      | 12     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 13     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 14     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 15     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 16     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 17     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 18     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 19     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 20     |            |      |                    |                    |                    |
|        |            |      |        |            |      |              |            |      | 21     |            |      |                    |                    |                    |



## Cite Us

```
```
