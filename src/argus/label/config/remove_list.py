patterns_to_remove = []
patterns_to_split = dict()

# patterns_to_remove.append(r'^[\w\s]+[0-9]+$')
patterns_to_remove.append(r"^[0-9.\-\+]+$")
patterns_to_remove.append(r"^[0-9.]+\s[\w\-\s]{2,20}$")
patterns_to_remove.append(r"^(FIGUUR.*)|(TABEL.*)|(Figuur.*)|(Tabel.*)$")
patterns_to_remove.append(r"^(HOOFDSTUK\s[0-9.]+)|(Hoofdstuk\s[0-9.]+)$")
patterns_to_remove.append(r"^(PARAGRAAF\s[0-9.]+)|(Paragraaf\s[0-9.]+)$")
patterns_to_remove.append(
    "Richtlijnendatabase\nMenu\nRichtlijnenNieuwsInstructiesOverContactInloggenCovid-19\nTerug naar zoekresultaten"
)
patterns_to_remove.append(
    "PDF\nDirect naar\nFederatie Medisch Specialisten\nOpleidingsetalage\nMedische vervolgopleidingen\nTOKIO Optimum Project\n"
)
patterns_to_remove.append(r"Over de Federatie\nWat is de Federatie\?\nWat doet de Federatie\?\n")
patterns_to_remove.append("Nuttige links\nDiensten & advies\nWerken bij\nGebruikersvoorwaarden\nPrivacyverklaring\n")
patterns_to_remove.append(
    r"Contact\nMercatorlaan 1200\n3528 BL Utrecht\n\(088\) 505 34 34\ninfo@demedischspecialist.nl\n"
)
patterns_to_remove.append("KvK Federatie Medisch Specialisten: 40483480")
patterns_to_remove.append("Privacyverklaring\nDisclaimer\n")
patterns_to_remove.append(r"Laatst beoordeeld:\n\d\d-\d\d-\d\d\d\d")
patterns_to_remove.append(r"Laatst geautoriseerd:\n\d\d-\d\d-\d\d\d\d")
patterns_to_remove.append("Bijlagen\n")
patterns_to_remove.append("PDF\n")
patterns_to_remove.append("Overslaan en naar de inhoud gaan\n")
patterns_to_remove.append(
    "Home\nlogo\nlogo-nhg-visual\nNHG-Richtlijnen\nZoektermen\nZoeken\nHome\nTotstandkoming NHG-Standaarden\nKennislacunes en huisartsgeneeskundig onderzoek\nVervallen richtlijnen\nOver NHG-richtlijnen\nContact\nExtra functies\nBreadcrumb\nHome"
)
patterns_to_remove.append(
    "KernboodschappenInleidingScopeAchtergrondenBegrippenEpidemiologieEtiologieRichtlijnen diagnostiekAnamneseEvaluatieRichtlijnen beleidVoorlichtingMedicamenteuze behandelingReferenties\n"
)
patterns_to_remove.append("Naar Samenvatting ›")
patterns_to_remove.append("Naar Volledige tekst ›")
patterns_to_remove.append("‹ Terug naar nomale weergave")
patterns_to_remove.append("Documenten\n")
patterns_to_remove.append("Printen\n")
patterns_to_remove.append("Volledig\n")
patterns_to_remove.append("E-mail\n")
patterns_to_remove.append("Facebook\n")
patterns_to_remove.append("Twitter\n")
patterns_to_remove.append("LinkedIn\n")
patterns_to_remove.append(r"Versie:\nVersie\s\d.\d")
patterns_to_remove.append("Versie:\nVersie")
patterns_to_remove.append(r"Revisiedatum\n[A-Za-z]{3,10} \d{4}")
patterns_to_remove.append(r"Auteur\(s\):\n[A-Za-z,.\- ]*\n")
patterns_to_remove.append(r"NHG-werkgroep\n[A-Za-z,.\- ]*\n")
patterns_to_remove.append(r"Revision status\n[A-Za-z ]*\n")
patterns_to_remove.append(r"KNR nummer\n[A-Z0-9]*\n")
patterns_to_remove.append("KNR nummer\n")
patterns_to_remove.append("NHG-Standpunten\n")
patterns_to_remove.append("NHG-Richtlijnen\n")
patterns_to_remove.append("LESA's\n")
patterns_to_remove.append("MDR'en\n")
patterns_to_remove.append("Thuisarts\n")
patterns_to_remove.append("Samenvatting\n")
patterns_to_remove.append("Hoofdtekst\n")
patterns_to_remove.append("Overslaan en naar de inhoud gaan\n")
patterns_to_remove.append("Search terms\n")
patterns_to_remove.append(
    "Diagnostiek\nSpoedzorg\nE-health\nKwaliteit & HRMO\nKetenzorg\nLevenseinde\nSOLK\nPatiënteninbreng\nSeksuele gezondheid\nCOVID-19\n"
)
patterns_to_remove.append(
    r"NHG-lid\nBent u NHG-lid\?Door in te loggen op Haweb krijgt u toegang tot de artikelen in de H&W website.\n"
)
patterns_to_remove.append(
    "H&W abonneeBij uw eerste bezoek aan de website, dient u zich opnieuw te registreren. Registreer hier.\nE-mail\nWachtwoord\nOnthoud mijn gegevens\nUw wachtwoord opnieuw instellenRegistreren als H&W-abonnee\n"
)
patterns_to_remove.append("Gebruikersmenu\nInloggen\n")
patterns_to_remove.append(
    "Hoofdnavigatie\nHome\nContact\nOver H&W\nVoor auteurs\nAdverteren\nH&W Podcast\nMedische banenbank\n"
)
patterns_to_remove.append(r"Gebruikersmenu \[mobile\]\nInloggen\n")
patterns_to_remove.append("PDF\n")
patterns_to_remove.append("DelenE-mail\nFacebook\nTwitter\nLinkedin\nReageren\n")
patterns_to_remove.append(r"Download dit nummer \(PDF\)\n")
patterns_to_remove.append(r"\d*\nreacties")
patterns_to_remove.append("Reacties\nEr zijn nog geen reacties\n")
patterns_to_remove.append(r"Door\n[A-Za-z\- ]\n")
patterns_to_remove.append(r"Gepubliceerd[0-9]{1,2} [A-Za-z]* \d{4}\n")
patterns_to_remove.append("Alle nummers\n")
patterns_to_remove.append("Artikelinfo\n")
patterns_to_remove.append(r"Editie\n[0-9a-z, ]*\n")
patterns_to_remove.append(r"Eerste paginanummer\n\d*\n")
patterns_to_remove.append(r"Laatste paginanummer\n\d*\n")
patterns_to_remove.append(r"Archief tijdschrift\n[0-9a-z, ]*\n")
patterns_to_remove.append(r"Jaargang\n\d*\n")

patterns_to_split["henw"] = (
    r"Huisarts en Wetenschap\nPrivacyverklaring\nAls eerste op de hoogte\nLinkedin\nTwitter\nRSS"
)
patterns_to_split["rd_richtlijn"] = (
    r"De Federatie Medisch Specialisten maakt gebruik van cookies\nDeze website van de Federatie Medisch Specialisten maakt gebruik van cookies, hiermee verzamelen we informatie over het gebruik van de website. Deze worden gebruikt voor analyse en verbetering van de website.\nIk wil meer informatie\nIk ga akkoord\nIk ga niet akkoord\n"
)
patterns_to_split["nhg"] = (
    r"Contact\nPrivacy- en cookiebeleid\nDisclaimer\nNaar NHG\.org\nNHG-Richtlijnen\nRichtlijnen voor de diagnostiek en behandeling van veelvoorkomende aandoeningen in de huisartsenpraktijk"
)
patterns_to_split["medi_wiki"] = r"^[\w\s]{1,50}$"
patterns_to_split["corpora"] = r"\n"
patterns_to_split["nvvc"] = r"\n"
patterns_to_split["nvr"] = r"\n"
patterns_to_split["nvog"] = r"\n"
patterns_to_split["nvmdl"] = r"\n"
patterns_to_split["ntvg"] = r"\n"
patterns_to_split["mantra"] = r"\n"
patterns_to_split["fms"] = r"\n"
patterns_to_split["dictionaries"] = r"\n"
patterns_to_split["books"] = r"\n"
patterns_to_split["dhd"] = r"\n"
patterns_to_split["dsm"] = (r"^[0-9]+\.[0-9]+\.[0-9]+\.?", 128, 120)
patterns_to_split["phd_summaries"] = r"\n"
patterns_to_split["www_scraping"] = r"\n"
patterns_to_split["misc"] = r"\n"
patterns_to_split["twnc"] = r"\n"
