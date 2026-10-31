# Feature contract

All named columns are required in CSVs. A blank value means missing; extra columns are ignored. Defaults are synthetic training medians/modes. Numeric bounds constrain serving and mark unsupported source values missing during training. They do not define a population validity guarantee.

| Source field | Meaning / units | Accepted values |
|---|---|---|
| `reading_score` | reading score | 0 to 100 |
| `writing_score` | writing score | 0 to 100 |
| `lunch` | lunch | free/reduced, standard |
| `test_preparation_course` | test preparation course | completed, none |
