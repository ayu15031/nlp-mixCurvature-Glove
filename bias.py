from utils import find_distance
import torch

weat_career = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
weat_family = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]
weat_arts = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture", "shakespeare"]
weat_science = ["science", "technology", "physics", "chemistry", "einstein", "nasa", "experiment", "astronomy"]

weat_male = ["he", "his", "man", "male", "boy", "son", "brother", "father", "uncle", "gentleman"]
weat_female = ["she", "her", "woman", "female", "girl", "daughter", "sister", "mother", "aunt", "lady"]

categories = ["career", "family", "arts", "science"]

def bias(word, emb_list, c):
  try:
    for emb in emb_list:
      emb[word]
  except:
    raise Exception("word not in embedding")
  male_count = 0
  female_count = 0
  male_similarity = 0
  female_similarity = 0
  for male_word in weat_male:
    try:
      for emb in emb_list:
        male_similarity += find_distance(emb[word], emb[male_word], c)
      male_count += 1
    except:
      print("male except")
      continue
  for female_word in weat_female:
    try:
      for emb in emb_list:
        female_similarity += find_distance(emb[word], emb[female_word], c)
      female_count += 1
    except:
      print(female_word)
      continue
  
  compute_bias = (male_similarity / male_count) - (female_similarity / female_count)
  return compute_bias

# input: category
def bias_category(category, emb_list, c):
  num_valid = 0
  total_bias = 0
  if category == "career":
    for word in weat_career:
      try:
        total_bias += bias(word, emb_list, c)
        num_valid += 1
      except:
        print(word)
        continue
  elif category == "family":
    for word in weat_family:
      try:
        total_bias += bias(word, emb_list, c)
        num_valid += 1
      except:
        print(word)
        continue
  elif category == "arts":
    for word in weat_arts:
      try:
        total_bias += bias(word, emb_list, c)
        num_valid += 1
      except:
        print(word)
        continue
  elif category == "science":
    for word in weat_science:
      try:
        total_bias += bias(word, emb_list, c)
        num_valid += 1
      except:
        print(word)
        continue
  else:
    raise Exception("category not valid")
  
  return total_bias / num_valid

def all_bias(emb_list, categories, c):
  biases = []
  for category in categories:
    try:
      biases.append(bias_category(category, emb_list, c))
    except:
      continue
  return biases

if __name__ == "__main__":
  emb_list = []
  for i in range(3):
    embeddings = {"king": torch.rand(20), "queen": torch.rand(20), "man": torch.rand(20), "women": torch.rand(20), "a": torch.rand(20), "women2": torch.rand(20), "women3": torch.rand(20)}
    emb_list.append(embeddings)
  c=0
  all_bias(emb_list, categories, c)