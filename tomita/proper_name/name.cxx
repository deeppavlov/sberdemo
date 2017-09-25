#encoding "utf8"
#GRAMMAR_ROOT ProperName

Name -> Word<gram="имя"> interp (ProperName.First) (Word<gram="отч"> interp (ProperName.Middle));
LastName -> Word<gram="фам"> interp (ProperName.Last);

ProperName -> (LastName) Name;
ProperName -> Name (LastName);