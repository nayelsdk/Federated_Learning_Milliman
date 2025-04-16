

###########################
# Exercice 4 
###########################

# Packages à installer pour cet exercice
# install.packages("haven")
# install.packages("sandwich")
# install.packages("dplyr")
library(haven)
library(sandwich)
library(dplyr)

# On charge le tableau de données (remplacer par votre propre chemin)
murder <- read_dta("/Users/Pro/Desktop/Enseignements - ENSAE/Econo 2/2024-2025/TD2/MURDER.DTA")
head(murder)
summary(murder)

# id : variable d'identification de l'État 
# year = t : désigne l'année de l'observation des informations 
# on dispose pour chaque état à chaque année, de son taux de meurtre (pour 100 000 personnes), du nombre d'exécution les 3 dernières années, du taux de chômage moyen l'anée considérée. 


    #### Question 2 ####

#install.packages("lmtest")
library(zoo)
library(lmtest)

# On va d'abord faire naïvement un MCO empilé sur l'ensemble des observations. 
pooled_ols <- lm(mrdrte ~ exec + unem + d90 + d93, data = murder)
# Le modèle inclut automatiquement une constante, donc on retire d87 de l'équation, 
# car d87, d90 et d93 sont liées linéairement. 

# Et on va aussi contrôler l'autocorrelation des résidus pour les observations d'un même État 
# Cela va juste changer la valeur des écarts-types. 

# Clustered erreurs standardes au niveau de la variable "id" 
# Cette commande calcule la matrice de variance-covariance robuste aux clusters
clustered_se <- vcovCL(pooled_ols, cluster = ~id)

# Affichage des paramètres du modèle avec les erreurs standards clusterisées
coeftest(pooled_ols, vcov = clustered_se)


# Nous regardons le coefficient devant la variable exec. 
# Nous voyons que, toutes choses égales par ailleurs, une exécution supplémentaire entraîne en moyenne 0,163 homicide supplémentaire pour 100 000 habitants.
# Mais les hypothèses pour la consistance de notre estimateur sont plutôt restrictives. 
# Qu'est-ce qu'on doit supposer ? 
# Pourquoi c'est probablement pas le cas ? 



    #### Question 3 ####

# On estime ici le modèle par différences premières. 

### MÉTHODE 1 : avec les variables delta à la main 
# L'estimateur par différences premières peut être construit à la main, en créant des variables 
# \deltaX_{t} = X_{t} - X{t-1}
# Elles ont déjà été créées dans le tableau (commencent par c) 

# On peut donc faire OLS avec ces variables déjà créées (on perd une date, donc on n'a plus que 2 dates,
# on en enlève une dans l'équation car elles sont parfaitement liées linéairement)
FD_model1 <- lm(cmrdrte ~ d93 + cexec + cunem, data = murder)
# Pareil pour les erreurs standard qu'avant : on a encore 2 dates pour 1 individu. 
clustered_se <- vcovCL(FD_model1, cluster = ~id)
# Affichage des paramètres du modèle avec les erreurs standards clusterisées
coeftest(FD_model1, vcov = clustered_se)

### MÉTHODE 2 : avec un package sur R et la déclaration de données de panel à R
# On peut aussi calculer le modèle différences premières avec le package plm
# Package plm : spécialisé dans l’estimation de modèles économétriques pour données de panel.
# install.packages("plm")
library(plm)

# On génère un indicateur de temps basé sur les conditions : 
# on créé la variable t qui est égale à 1, 2 ou 3 selon l'année de l'observation.  
# t = 1 si on est en 87, 2 si 90 et 3 si 93
murder <- murder %>%
  mutate(t = 1 + (year >= 90) + (year >= 93))

# On déclare qu'on est dans une structure de panel. On doit dire quelle variable est i, et déclarer t. 
# i : id = identifiant de l'État ; et t : l'indicateur temporel créé juste avant 
# la commande pdata.frame convertit un dataframe classique en un dataframe panel
pmurder <- pdata.frame(murder, index = c("id", "t"))

# Régression de first-différence : comme on a déclaré la structure de panel pas besoin ici de passer par les variables cmrdrte, etc. 
# On précise juste quel estimateur on veut avec model = "fd" pour first-difference. 

FD_model2 <- plm(mrdrte ~ exec + unem + d93, 
                 data = pmurder, model = "fd")

# Toujours des erreurs standardes clusterisées au niveau de la variable id
clustered_se <- vcovHC(FD_model2, method = "arellano", type = "HC2", cluster = "group")
# On a rajouté ici un type de méthode de calcul approprié pour les données de panel
# + type HC2 souvent utilisé quand on a des petits échantillons
summary_FD_model2 <- coeftest(FD_model2, vcov = clustered_se)
print(summary_FD_model2)


# Dans les deux cas (à la main ou à l'aide de la commande d(), on estime un coefficient négatif : 
# toutes choses égales par ailleurs, une exécution additionnelle réduit le taux de meurtre pour 10^4 habitants de 0.115. 
# Le résultat est significatif à 1%)

# Pour la question plus générale : est-ce que la peine de mort dissuade de commettre un meurtre ? 
# ça peut être le cas si nos estimateurs sont bien consistants. 
# Pour ça, on a besoin que l'hypothèse d'exogénéité stricte soit vérifiée. 



    #### Question 5 ####

# On a remis en cause l'hypothèse d'exogénéité stricte qui nous permettait d'interpréter le précédent 
# coeff comme l'estimation d'un effet du nombre d'exécution sur le nombre de meurtres. 
# Maintenant, avec l'hypothèse d'exogénéité faible, on a besoin d'un instrument. 

# Load necessary packages
#install.packages("AER") # For 2SLS regression

library(AER)
detach("package:dplyr", unload = TRUE)

# Step 1: On créé les variables différenciées (delta)
pmurder$d_mrdrte <- diff(pmurder$mrdrte)
pmurder$d_exec <- diff(pmurder$exec)
pmurder$d_unem <- diff(pmurder$unem)

# Step 2: On créé les instruments (lag of exec and lag of d_exec)
pmurder$lag_exec <- lag(pmurder$exec, 1)
pmurder$lag_d_exec <- lag(diff(pmurder$exec), 1)

# INSTRUMENT 1 : lag of exec
iv_model1 <- plm(mrdrte ~ exec + unem + d93 | lag_exec + unem + d93, model = "fd", method = "iv", data = pmurder)
summary(iv_model1, vcov = vcovHC)

# On regarde le first stage
#install.packages("estimatr")
library(dplyr)
library(estimatr)

first_stage1 <- lm(d_exec ~ lag_exec + d_unem + d93, se_type = "stata", data = pmurder %>% filter(year != 87))
clustered_se <- vcovCL(first_stage1, cluster = ~id)
coeftest(first_stage1, vcov = clustered_se)

# Ou bien reprendre le tableau murder de départ, et créer la variable lag 
first_stage1 <- lm(cexec ~ l_exec + cunem + d93, se_type = "stata", data = murder %>% group_by(id) %>% mutate(l_exec = lag(exec)) %>% ungroup() %>% filter(year != 87))
clustered_se <- vcovCL(first_stage1, cluster = ~id)
coeftest(first_stage1, vcov = clustered_se)

# Le coef devant l_exec (-0.12) n'est pas significatif -> faible instrument 

# INSTRUMENT 2 : lag of d_exec

### Notice we end up with cross sectional data with 51 observations

iv_model2 <- ivreg(d_mrdrte ~ d_exec + d_unem | lag_d_exec + d_unem, data = pmurder %>% filter(year == 93))
summary(iv_model2)

# Check 1st stage
first_stage2 <- lm_robust(d_exec ~ lag_d_exec + d_unem, se_type = "stata", data = pmurder %>% filter(year == 93))
summary(first_stage2)

# Maintenant le coeff devant lag_d_exec dans la first stage est significatif. Déjà mieux. 


# On affiche les résultats 
summary(iv_model1)
summary(iv_model2)


# Ainsi, on lit les résultats de notre 2e modèle instrumental : 
# Toutes choses égales par ailleurs, une exécution additionnelle mène à une réduction de 0.100 du taux de meurtre. 
# Mais le coefficient n'est pas significatif, 
# donc on ne peut pas rejeter l'hypothèse nulle selon laquelle la peine de mort n'a aucun effet dissuasif sur les meurtres.  



    #### Question 6 ####
# On va comparer le Texas aux autres États. 
# On fait la même chose qu'avant sans le Texas. 

murder %>%
  mutate(istexas = (state == 'TX')) %>%
  group_by(istexas, t) %>%
  summarise(avg.exec = mean(exec))

# Cette commande précédente créée une variable binaire = 1 si l'État considéré est le Texas. 
# Puis elle calcule pour chacun des 2 groupes Texas et non Texas la moyenne du nombre d'exécutions. 
# Que remarque-t-on ? 

pmurder_notx = pmurder[pmurder$state != 'TX', ]
# On créé ici un tableau dans le Tewas.  


# First-difference regression (FD regression) without Texas
FD_model2b <- plm(mrdrte ~ exec + unem + d93,
                  data = pmurder_notx, model = "fd")

# Clustered standard errors at the id level
clustered_se <- vcovHC(FD_model2b, method = "arellano", type = "HC2", cluster = "group")

# Summary of the model with clustered standard errors
summary_FD_model2b <- coeftest(FD_model2b, vcov = clustered_se)

# Previous FD with Texas
print(summary_FD_model2)
# New FD without Texas
print(summary_FD_model2b)

# Qu'observe-t-on ? 

# IV without Texas
first_stage1b <- lm_robust(d_exec ~ lag_exec + d_unem, se_type = "HC2",data = pmurder %>% filter(year == 93, state != 'TX'))
summary(first_stage1) # previous with Texas
summary(first_stage1b) # new without Texas
#
first_stage2b <- lm_robust(d_exec ~ lag_d_exec + d_unem, se_type = "HC2" , data = pmurder %>% filter(year == 93, state != 'TX'))
summary(first_stage2) # previous with Texas
summary(first_stage2b) # new without Texas
# Perte de significativité dans la first stage : l'instrument devient faible. 

iv_model1b <- ivreg(d_mrdrte ~ d_exec + d_unem | lag_exec + d_unem, data = pmurder_notx)
iv_model2b <- ivreg(d_mrdrte ~ d_exec + d_unem | lag_d_exec + d_unem, data = pmurder_notx)

# Display the summary of the 2SLS regression
# IV1: lag exec
summary(iv_model1)
summary(iv_model1b)
# IV2: lag d_exec
summary(iv_model2)
summary(iv_model2b)

## En omettant le Texas, la valeur du coefficient dans le modèle en première différence ne change pas beaucoup 
# mais les erreurs standard du coefficient associé à d.exec explosent. 
# Nous ne pouvons pas rejeter l'hypothèse selon laquelle la peine de mort n'a pas d'effet dissuasif sur les homicides. 
# Lorsque nous examinons la régression instrumentale où l'instrument est la valeur retardée de cexec, 
# l'instrument devient faible. Lorsque nous examinons la base de données, nous constatons que, par rapport à d'autres États, 
# le Texas est une aberration. Le nombre d'exécutions est important et varie beaucoup dans le temps. 
# Lorsque nous supprimons cet État, nous réduisons la variation de la valeur retardée de cexec, 
# ce qui affecte considérablement les erreurs standardes.



