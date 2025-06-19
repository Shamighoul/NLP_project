# NLP_project
Project for MTS-NLP cources

# AI-Powered Recipe Generation

## Abstract

Recipe generation from recipe titles only is currently unsolved, as state of the art models require both recipe titles and ingredients lists for instruction generation (Lee et al., 2020). This project investigates if a number of different architectures such as Long Short-Term Memory (LSTM) encoder-decoders, LSTM decoders, or Transformer-based decoders, can produce meaningful ingredient lists when given recipe titles only. The recipe titles and generated ingredients are then passed into an existing recipe instruction generation framework to produce cooking instructions (Liu et al., 2022).

## Introduction

In today's multicultural and diverse world, food preferences and dietary restrictions play a crucial role in people's daily lives. Religious dietary laws, such as _Halal_ (Islam), _Kosher_ (Judaism), _vegetarianism_ (Hinduism, Buddhism, Lenten), and other faith-based restrictions, significantly influence what individuals can or cannot eat. However, finding suitable recipes that align with both available ingredients and religious requirements can be challenging.

This problem is relevant for several reasons:

1. **Cultural and Religious Sensitivity**  
   Many people strictly follow dietary laws due to their faith, and violating these rules can be offensive or unacceptable.

2. **Reducing Food Waste**  
   By suggesting recipes based on available ingredients, the system encourages efficient use of food resources.

3. **Convenience & Personalization**  
   Traditional recipe apps often lack filters for religious constraints, forcing users to manually check each ingredient. An AI that automates this process saves time and improves accessibility.

4. **Global Applicability**  
   With increasing multicultural interactions (travel, migration, international cuisine), such a tool can assist both individuals and businesses (restaurants, catering) in offering compliant meal options.
