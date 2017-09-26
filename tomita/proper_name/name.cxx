#encoding "utf8"
#GRAMMAR_ROOT ProperName

Name -> Word<gnc-agr[1],gram="имя"> interp (ProperName.First) (Word<gnc-agr[1],gram="отч"> interp (ProperName.Middle));
LastName -> Word<gram="фам"> interp (ProperName.Last);

ProperName -> (LastName) Name;
ProperName -> Name (LastName);