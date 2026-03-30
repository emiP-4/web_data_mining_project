"""
Lab Session: Reasoning (SWRL)
Course: Web Mining & Semantics

Step 3.2 - Olympics Knowledge Base Reasoner
──────────────────────
This script uses Owlready2 and the Pellet reasoner to deduce new 
knowledge using a Semantic Web Rule Language (SWRL) rule.

Usage:
  pip install owlready2
  python src/reasoning/olympics_reasoner.py
"""

from owlready2 import get_ontology, Thing, ObjectProperty, Imp, sync_reasoner_pellet

def main():
    print("=== Olympics Knowledge Graph: SWRL Reasoning ===\n")
    
    # 1. Create a namespace/ontology for our Olympics domain
    # (If your previous steps exported an OWL/XML file, you could load it 
    # directly here via get_ontology("file://path/to/file.owl").load())
    onto_path = "http://olympics.kg/ontology.owl"
    onto = get_ontology(onto_path)

    with onto:
        # --- 2. Define Ontology Schema (Classes and Properties) ---
        class Person(Thing): pass
        class Athlete(Person): pass
        class Medal(Thing): pass
        
        # The target class we want to INFER (currently empty)
        class GoldMedalist(Athlete): pass 

        class wonMedal(ObjectProperty):
            domain = [Athlete]
            range = [Medal]

        # --- 3. Populate KB Instances (Mocking data from extracted_knowledge.csv) ---
        # Define specific medal entities
        gold_medal = Medal("Olympic_gold_medal")
        silver_medal = Medal("Olympic_silver_medal")

        # Define athletes and their extracted relationships
        usain = Athlete("Usain_Bolt")
        usain.wonMedal.append(gold_medal)

        simone = Athlete("Simone_Biles")
        simone.wonMedal.append(gold_medal)
        
        # A control case: an athlete who didn't win gold
        john = Athlete("John_Doe")
        john.wonMedal.append(silver_medal)

        print("[INFO] Graph loaded. Current explicitly defined Gold Medalists:")
        # This will be empty because we haven't assigned anyone to this class explicitly!
        current_medalists = list(GoldMedalist.Instances)
        print(f"       {current_medalists if current_medalists else 'None'}\n")

        # --- 4. Define the SWRL Rule ---
        print("[INFO] Injecting SWRL Rule: ")
        print("       Athlete(?a) ^ wonMedal(?a, Olympic_gold_medal) -> GoldMedalist(?a)\n")
        
        rule = Imp()
        rule.set_as_rule(
            "Athlete(?a), wonMedal(?a, Olympic_gold_medal) -> GoldMedalist(?a)"
        )

    # --- 5. Run the Pellet Reasoner ---
    print("[INFO] Running Pellet reasoner... (inferring new triples)")
    # sync_reasoner_pellet automatically detects the SWRL rule and applies it
    with onto:
        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)

        # --- 6. Output the Results ---
        print("\n=== Reasoner Output ===")
        print("[SUCCESS] Inferred Gold Medalists:")
        
        # Query the class again to see the inferred entities
        for individual in GoldMedalist.Instances:
            print(f" -> {individual.name}")

        print("\n[VERIFICATION]")
        is_john_gold = "GoldMedalist" in [c.__name__ for c in john.INDIRECT_classes]
        print(f"Did the reasoner accidentally classify John_Doe (Silver)? : {is_john_gold}")
        print("================================================")

if __name__ == "__main__":
    main()