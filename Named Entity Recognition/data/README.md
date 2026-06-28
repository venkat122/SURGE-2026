EkStep Indian Legal NER Dataset
The **EkStep** dataset consists of annotated Indian Supreme Court judgments with 14 Named Entity types. It is the standard benchmark for Indian Legal NER research.

| Entity | Description | Example |
|---|---|---|
| COURT | Name of the court | Supreme Court of India |
| PETITIONER | Petitioning party | State of Maharashtra |
| RESPONDENT | Responding party | Rajesh Kumar |
| JUDGE | Name of the judge | Justice R.F. Nariman |
| LAWYER | Name of the advocate | Mr. Shekhar Naphade |
| DATE | Dates mentioned | 12th March 2019 |
| ORG | Organizations | Central Bureau of Investigation |
| GPE | Geo-political entities | New Delhi |
| STATUTE | Name of statutes | Indian Penal Code |
| PROVISION | Specific sections/articles | Section 302 |
| PRECEDENT | Referenced case names | Kedar Nath Singh v. State of Bihar |
| CASE_NUMBER | Case identifiers | Criminal Appeal No. 1234/2020 |
| WITNESS | Witness names | PW-1 Ramesh |
| OTHER_PERSON | Other named persons | Mahatma Gandhi |

| Split | Documents | Estimated Sentences |
|---|---|---|
| Train | ~3,000+ | ~9,000–10,000 |
| Dev | ~1,074 | ~1,500–2,000 |
| Test | ~1,000+ | ~1,500–2,000 |

The dataset files are too large (~67MB+) to include directly in the repository. Download them from the official source:

- **Train**: https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TRAIN.zip
- **Dev**: https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_DEV.zip
- **Test**: https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/NER/NER_TEST.zip

After downloading, extract the ZIP files into this directory:
```
data/
├── train/    # Extract NER_TRAIN.zip here
├── dev/      # Extract NER_DEV.zip here
└── test/     # Extract NER_TEST.zip here
```

```
