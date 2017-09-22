#encoding "utf8"
#GRAMMAR_ROOT House

StreetW ->  Word<kwset=["тип_улицы"]> interp (Street.Descr);
StreetSokr -> Word<kwset=["сокр_тип_улицы"]> interp (Street.Descr::not_norm) {outgram="сокр"} ;

StreetDescr -> StreetW | StreetSokr;


StreetNameNoun -> (Adj<gnc-agr[1]>) Word<gnc-agr[1],rt> (Word<gram="род">);

NumberW_1 -> AnyWord<wff=/[1-9][0-9]?-?((ый)|(ий)|(ой)|й)/> {outgram="муж,ед,им"};
NumberW_2 -> AnyWord<wff=/[1-9][0-9]?-?((ая)|(яя)|(ья)|я)/> {outgram="жен,ед,им"};
NumberW_3 -> AnyWord<wff=/[1-9][0-9]?-?((ее)|(ье)|(ое)|е)/> {outgram="сред,ед,им"};

NumberW -> NumberW_1 | NumberW_2 | NumberW_3;

StreetDescr -> NumberW<gnc-agr[1]> interp (Street.LineNumber) StreetW<gnc-agr[1]>;

StreetNameAdj -> Adj Adj*;
StreetNameAdj -> NumberW<gnc-agr[1]> Adj<gnc-agr[1]>;


Street -> StreetDescr StreetNameNoun<gram="род"> interp (Street.StreetName::not_norm);
Street -> StreetDescr StreetNameNoun<gram="им"> interp (Street.StreetName::not_norm);

Street -> StreetNameAdj<gnc-agr[1]> interp (Street.StreetName) StreetW<gnc-agr[1]>;
Street -> StreetNameAdj interp (Street.StreetName) StreetSokr;
Street -> StreetW<gnc-agr[1]> StreetNameAdj<gnc-agr[1]> interp (Street.StreetName);
Street -> StreetSokr StreetNameAdj interp (Street.StreetName);



DomDeskr -> 'д' | 'дом';
Dom -> (DomDeskr) AnyWord<wff="\\d+((к|/)\\d*)?">;
Dom -> AnyWord<wff="(д|(дом))?\\d+((к|/)\\d*)?">;


House -> Street (Dom interp (Street.House));