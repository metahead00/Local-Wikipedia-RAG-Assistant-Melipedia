# Recommendations

1. Retrieved Text
Sometimes the retrieved text contains non-latin letters and this affects the embedding, usually having a negative effect. A preprocessing to the retrieved text would be safer than direct embedding.

2. Wikipedia Rate Limit
If the fetching from wikipedia is fast, we get an 500 error due to rate limits. To overcome this, sleep lines were added in case of such error. If more websites will be fetched, more rigorous solutions are needed to resolve this issue.

3. Stream LLM
A streaming interface would look better since the LLM works slow on CPU and sometimes it looks like the UI is frozen because of it.