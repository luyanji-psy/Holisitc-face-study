library(lme4)
library(lmerTest)
library(optimx)

# load the data
load("df_lmm.RData")

######### max model #########
# message("Fitting glmm_resp_max...")
# glmm_resp_max <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Congruency * Alignment * SameDifferent | Participant), # Con_Ali_Sam
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_max, file = "Resp_lmm_max.RData")
# message("max is finished.")


######### zcp model #########
# message("Fitting glmm_resp_zcp")
# glmm_resp_zcp <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Ali + Con_Sam + Ali_Sam +
#              Cue_Con_Ali + Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam + 
#              Cue_Con_Ali_Sam || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_zcp, file = "Resp_lmm_zcp.RData")
# message("zcp is finished.")


######## rdc model #########
# message("Fitting glmm_resp_rdc")
# glmm_resp_rdc <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam || Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# 
# save(glmm_resp_rdc, file = "Resp_lmm_rdc.RData")
# message("rdc is finished.")


######### etd model #########
# message("Fitting glmm_resp_etd")
# glmm_resp_etd <- glmm_resp_etd <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Con_C + Ali_C + Sam_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# 
# save(glmm_resp_etd, file = "Resp_lmm_etd.RData")
# message("etd is finished.")


######### update etd model as etd1
# load("Resp_lmm_etd.RData")
# ss <- getME(glmm_resp_etd, c("theta","fixef"))
# glmm_resp_etd1 <- update(
#     glmm_resp_etd, start=ss,
#     control=glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                          optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE)))
# 
# save(glmm_resp_etd1, file = "Resp_lmm_etd1.RData")
# message("etd1 is finished.")


######### update etd model as etd2
# glmm_resp_etd2 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Ali_C + Sam_C + # Con_C + 
#              Cue_Con + Cue_Ali + Cue_Sam + Con_Sam + # Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd2, file = "Resp_lmm_etd2.RData")
# message("etd2 is finished.")


######### update etd model as etd3
# glmm_resp_etd3 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Ali_C + Sam_C + # Con_C + 
#              Cue_Ali + Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd3, file = "Resp_lmm_etd3.RData")
# message("etd3 is finished.")

######### update etd model as etd4
# glmm_resp_etd4 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Ali + Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd4, file = "Resp_lmm_etd4.RData")
# message("etd4 is finished.")

######### update etd model as etd5
# glmm_resp_etd5 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
#              Cue_Con_Sam + Cue_Ali_Sam + Con_Ali_Sam +# Cue_Con_Ali + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd5, file = "Resp_lmm_etd5.RData")
# message("etd5 is finished.")

######### update etd model as etd6
# glmm_resp_etd6 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
#              Cue_Con_Sam + Con_Ali_Sam +# Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd6, file = "Resp_lmm_etd6.RData")
# message("etd6 is finished.")

######### update etd model as etd7
# glmm_resp_etd7 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (0 + Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
#              Cue_Con_Sam + Con_Ali_Sam +# Cue_Con_Ali + Cue_Ali_Sam + 
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd7, file = "Resp_lmm_etd7.RData")
# message("etd7 is finished.")

######### update etd model as etd8
# glmm_resp_etd8 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (0 + Cue_C + Sam_C + # Con_C + Ali_C + 
#              Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd8, file = "Resp_lmm_etd8.RData")
# message("etd8 is finished.")

######### update etd model as etd9
# glmm_resp_etd9 <- glmer(
#     Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
#         (0 + Sam_C + # Con_C + Ali_C +Cue_C + 
#              Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
#              Cue_Con_Sam + # Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +
#              Cue_Con_Ali_Sam | Participant),
#     family = binomial(link = "probit"),
#     data = df_lmm,
#     control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
#                            optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
# )
# save(glmm_resp_etd9, file = "Resp_lmm_etd9.RData")
# message("etd9 is finished.")

######### update etd model as etd10
glmm_resp_etd10 <- glmer(
    Resp ~ Cue * Congruency * Alignment * SameDifferent + Probability + 
        (0 + Sam_C + # Con_C + Ali_C +Cue_C + 
             Cue_Sam + Con_Sam + # Cue_Con + Con_Ali + Ali_Sam + Cue_Ali + 
             Cue_Con_Sam  # + Cue_Con_Ali + Cue_Ali_Sam + Con_Ali_Sam +
         | Participant), # Cue_Con_Ali_Sam 
    family = binomial(link = "probit"),
    data = df_lmm,
    control = glmerControl(optimizer = "optimx", # calc.derivs = FALSE,
                           optCtrl = list(method = "nlminb", starttests = FALSE, kkt = FALSE))
)
save(glmm_resp_etd10, file = "Resp_lmm_etd10.RData")
message("etd10 is finished.")




