def count_vowels(text):
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if vowels)
print (count_vowels("Heeellooo, world"))