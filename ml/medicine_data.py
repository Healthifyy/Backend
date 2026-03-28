MEDICINE_DATA = {
  "Fungal infection": {
    "otc_medicines": ["Clotrimazole cream", "Miconazole powder"],
    "prescription_medicines": ["Fluconazole", "Itraconazole"],
    "avoid": ["Steroids without antifungal"],
    "category": "antifungal"
  },
  "Allergy": {
    "otc_medicines": ["Cetirizine 10mg", "Loratadine 10mg", "Chlorpheniramine"],
    "prescription_medicines": ["Montelukast", "Fexofenadine"],
    "avoid": ["Aspirin if aspirin-sensitive"],
    "category": "antihistamine"
  },
  "GERD": {
    "otc_medicines": ["Antacids (Gelusil, Digene)", "Omeprazole 20mg", "Pantoprazole 40mg"],
    "prescription_medicines": ["Rabeprazole", "Esomeprazole"],
    "avoid": ["NSAIDs", "Alcohol", "Spicy food"],
    "category": "antacid/PPI"
  },
  "Drug Reaction": {
    "otc_medicines": ["Cetirizine for mild rash"],
    "prescription_medicines": ["Prednisolone", "Epinephrine for severe reaction"],
    "avoid": ["The causative drug — STOP immediately"],
    "category": "emergency"
  },
  "Peptic ulcer diseae": {
    "otc_medicines": ["Antacids", "Pantoprazole 40mg"],
    "prescription_medicines": ["Omeprazole + Clarithromycin + Amoxicillin (H.pylori triple therapy)"],
    "avoid": ["Aspirin", "Ibuprofen", "NSAIDs", "Alcohol"],
    "category": "PPI/antibiotic"
  },
  "AIDS": {
    "otc_medicines": [],
    "prescription_medicines": ["ART (Antiretroviral Therapy) — consult specialist immediately"],
    "avoid": ["Self-medication", "Unprotected contact"],
    "category": "antiviral"
  },
  "Diabetes": {
    "otc_medicines": ["Glucose tablets for hypoglycemia"],
    "prescription_medicines": ["Metformin 500mg/1000mg", "Glipizide", "Insulin"],
    "avoid": ["High sugar foods", "Steroids without monitoring"],
    "category": "antidiabetic"
  },
  "Gastroenteritis": {
    "otc_medicines": ["ORS (Oral Rehydration Solution)", "Zinc supplements", "Probiotics"],
    "prescription_medicines": ["Norfloxacin", "Metronidazole if parasitic"],
    "avoid": ["Dairy", "Fatty foods", "Antidiarrheals in children under 2"],
    "category": "rehydration/antibiotic"
  },
  "Bronchial Asthma": {
    "otc_medicines": ["Salbutamol inhaler (rescue)"],
    "prescription_medicines": ["Budesonide inhaler (controller)", "Montelukast", "Ipratropium"],
    "avoid": ["Beta-blockers", "Aspirin", "NSAIDs", "Dust and smoke"],
    "category": "bronchodilator/steroid"
  },
  "Hypertension": {
    "otc_medicines": [],
    "prescription_medicines": ["Amlodipine 5mg", "Losartan 50mg", "Atenolol 50mg", "Hydrochlorothiazide"],
    "avoid": ["High salt diet", "NSAIDs", "Decongestants"],
    "category": "antihypertensive"
  },
  "Migraine": {
    "otc_medicines": ["Paracetamol 500mg", "Ibuprofen 400mg", "Aspirin 600mg"],
    "prescription_medicines": ["Sumatriptan", "Topiramate (preventive)", "Amitriptyline"],
    "avoid": ["Bright lights", "Strong smells", "Skipping meals", "Overuse of painkillers"],
    "category": "analgesic/triptan"
  },
  "Malaria": {
    "otc_medicines": ["Paracetamol for fever"],
    "prescription_medicines": ["Artemether-Lumefantrine (AL)", "Chloroquine (if sensitive)", "Primaquine for P.vivax"],
    "avoid": ["Self-diagnosis", "Incomplete treatment course"],
    "category": "antimalarial"
  },
  "Chicken pox": {
    "otc_medicines": ["Calamine lotion", "Cetirizine for itch", "Paracetamol for fever"],
    "prescription_medicines": ["Acyclovir 800mg 5x daily (if severe or adult)"],
    "avoid": ["Aspirin in children (Reye syndrome risk)", "Scratching lesions", "School/public contact"],
    "category": "antiviral/symptomatic"
  },
  "Dengue": {
    "otc_medicines": ["Paracetamol for fever", "ORS for hydration"],
    "prescription_medicines": ["IV fluids if platelet low", "Platelet transfusion if critical"],
    "avoid": ["Aspirin", "Ibuprofen", "NSAIDs — increase bleeding risk"],
    "category": "supportive care"
  },
  "Typhoid": {
    "otc_medicines": ["Paracetamol for fever"],
    "prescription_medicines": ["Azithromycin 500mg x 7 days", "Cefixime 200mg BD", "Ciprofloxacin (if sensitive)"],
    "avoid": ["Unpurified water", "Outside food", "Incomplete antibiotic course"],
    "category": "antibiotic"
  },
  "Hepatitis A": {
    "otc_medicines": ["Paracetamol (low dose) for discomfort"],
    "prescription_medicines": ["Supportive care only", "No specific antiviral needed"],
    "avoid": ["Alcohol", "Paracetamol high dose", "Fatty food"],
    "category": "supportive"
  },
  "Hepatitis B": {
    "otc_medicines": [],
    "prescription_medicines": ["Tenofovir", "Entecavir", "Interferon alfa (specialist)"],
    "avoid": ["Alcohol", "Raw seafood", "Hepatotoxic medications"],
    "category": "antiviral"
  },
  "Tuberculosis": {
    "otc_medicines": ["Paracetamol for fever"],
    "prescription_medicines": ["HRZE regimen — 6 month DOTS (Isoniazid + Rifampicin + Pyrazinamide + Ethambutol)"],
    "avoid": ["Alcohol", "Missing doses — causes resistance"],
    "category": "antibiotic (DOTS)"
  },
  "Common Cold": {
    "otc_medicines": ["Paracetamol 500mg", "Cetirizine 10mg", "Steam inhalation", "Honey + ginger"],
    "prescription_medicines": ["Not needed usually", "Amoxicillin only if secondary bacterial infection"],
    "avoid": ["Antibiotics (viral)", "Cold drinks"],
    "category": "symptomatic"
  },
  "Pneumonia": {
    "otc_medicines": ["Paracetamol for fever"],
    "prescription_medicines": ["Amoxicillin-Clavulanate 625mg", "Azithromycin 500mg", "Ceftriaxone IV (if severe)"],
    "avoid": ["Self-stopping antibiotics early", "Cough suppressants"],
    "category": "antibiotic"
  },
  "Heart attack": {
    "otc_medicines": ["Aspirin 325mg — chew immediately if suspected"],
    "prescription_medicines": ["Call 108 immediately", "Nitroglycerin (if prescribed)", "Clopidogrel", "Heparin IV"],
    "avoid": ["Delay", "Physical exertion", "NSAIDs"],
    "category": "EMERGENCY — call 108"
  },
  "Acne": {
    "otc_medicines": ["Benzoyl peroxide 2.5-5% gel", "Salicylic acid face wash", "Clindamycin gel topical"],
    "prescription_medicines": ["Isotretinoin (severe, specialist)", "Doxycycline 100mg"],
    "avoid": ["Picking/squeezing", "Heavy oil-based products", "Sun exposure without SPF"],
    "category": "topical/antibiotic"
  },
  "Urinary tract infection": {
    "otc_medicines": ["Cranberry juice", "Plenty of water (3L/day)"],
    "prescription_medicines": ["Nitrofurantoin 100mg x 5 days", "Trimethoprim 200mg x 7 days", "Ciprofloxacin 500mg x 3 days"],
    "avoid": ["Holding urine", "Tight synthetic underwear"],
    "category": "antibiotic"
  },
  "Arthritis": {
    "otc_medicines": ["Ibuprofen 400mg", "Paracetamol 500mg", "Warm compress"],
    "prescription_medicines": ["Methotrexate (RA)", "Hydroxychloroquine", "Sulfasalazine"],
    "avoid": ["Cold weather without protection", "Heavy lifting", "Smoking"],
    "category": "DMARD/analgesic"
  },
  "Impetigo": {
    "otc_medicines": ["Mupirocin ointment topical"],
    "prescription_medicines": ["Flucloxacillin 250mg x 7 days", "Cefalexin 250mg x 7 days"],
    "avoid": ["Sharing towels/clothing", "Touching/scratching lesions", "School contact until healed"],
    "category": "antibiotic (topical/oral)"
  },
  "Jaundice": {
    "otc_medicines": [],
    "prescription_medicines": ["Ursodeoxycholic acid", "Antiviral if hepatitis B/C"],
    "avoid": ["Alcohol", "Paracetamol", "Fatty food", "All hepatotoxic drugs"],
    "category": "hepatic"
  },
  "Hypothyroidism": {
    "otc_medicines": [],
    "prescription_medicines": ["Levothyroxine 25-100mcg (fasting, morning)"],
    "avoid": ["Calcium/iron supplements within 4 hrs of dose", "Soy products", "Missing doses"],
    "category": "thyroid hormone"
  },
  "Hypoglycemia": {
    "otc_medicines": ["Glucose tablets", "Sugar water", "Fruit juice — take IMMEDIATELY"],
    "prescription_medicines": ["Glucagon injection (severe)", "Dextrose IV (hospital)"],
    "avoid": ["Skipping meals", "Excess insulin", "Alcohol on empty stomach"],
    "category": "glucose replacement — URGENT"
  },
  "Psoriasis": {
    "otc_medicines": ["Coal tar shampoo", "Moisturizers (petroleum jelly)", "Salicylic acid cream"],
    "prescription_medicines": ["Betamethasone cream", "Calcipotriol", "Methotrexate (severe)"],
    "avoid": ["Stress", "Smoking", "Alcohol", "Skin trauma"],
    "category": "topical steroid/immunomodulator"
  }
}

def get_medicine_info(disease_name: str) -> dict:
    if disease_name in MEDICINE_DATA:
        return MEDICINE_DATA[disease_name]
    for key in MEDICINE_DATA:
        if key.lower() == disease_name.lower():
            return MEDICINE_DATA[key]
    return {
        "otc_medicines": [],
        "prescription_medicines": ["Consult a doctor for medication advice"],
        "avoid": [],
        "category": "consult doctor"
    }
