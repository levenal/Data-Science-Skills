# Objective
Determine which skills are most frequently desired in job descriptions for data scientists.

# Procedure
1. Scrape Indeed.com for job descriptions using search terms of "data scientist" and a few other terms for distinction
2. Perform tfidf to identify terms (n-grams up to 5?) most strongly associated with "data scientist" as opposed to other job titles
3. Count number of "data scientist" positions that contain those terms
4. Sort terms by frequency
5. Visualize results
6. Search top terms to find other positions that require similar skills, find new job title search terms
7. Repeat for other data science related titles?
8. Implement search filtering by company NAICS codes?

This project is inspired by https://github.com/yuanyuanshi/Data_Skills
