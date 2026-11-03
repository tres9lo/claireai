# Corpus manifests

Before training, add one tracked JSON manifest per corpus. Include the source,
retrieval date, copyright/licence terms, redistribution status, exact files,
processing steps, and a contact or reference for the license. Never commit raw
corpus text or personal data. The trainer stores SHA-256 hashes in each model's
checkpoint manifest so a run can be tied to the files it used.
