"""
Lab Session: Reasoning (SWRL)
Course: Web Mining & Semantics

Step 3.1 - Baseline Reasoning on family.owl
──────────────────────
This script uses Owlready2 to demonstrate basic SWRL reasoning
by inferring the 'hasUncle' relationship.

Usage:
  python src/reasoning/family_reasoner.py
"""

from owlready2 import get_ontology, Thing, ObjectProperty, Imp, sync_reasoner_pellet

def main():
    print("=== SWRL Reasoning: family.owl ===\n")
    
    # Create a temporary ontology for the family domain
    onto = get_ontology("http://example.org/family.owl")

    with onto:
        # 1. Define the Schema (Classes and Properties)
        class Person(Thing): pass

        class hasParent(ObjectProperty):
            domain = [Person]
            range = [Person]

        class hasBrother(ObjectProperty):
            domain = [Person]
            range = [Person]

        class hasUncle(ObjectProperty):
            domain = [Person]
            range = [Person]

        # 2. Populate the Knowledge Base (The Family Tree)
        john = Person("John")
        mary = Person("Mary")
        robert = Person("Robert")

        # John's parent is Mary. Mary's brother is Robert.
        john.hasParent.append(mary)
        mary.hasBrother.append(robert)

        print("[INFO] Initial facts:")
        print(f" - John's parent: {[p.name for p in john.hasParent]}")
        print(f" - Mary's brother: {[b.name for b in mary.hasBrother]}")
        print(f" - John's uncle(s) BEFORE reasoning: {[u.name for u in john.hasUncle]}\n")

        # 3. Define the SWRL Rule
        print("[INFO] Injecting SWRL Rule:")
        print("       Person(?x) ^ hasParent(?x, ?y) ^ hasBrother(?y, ?z) -> hasUncle(?x, ?z)\n")
        
        rule = Imp()
        rule.set_as_rule(
            "Person(?x), hasParent(?x, ?y), hasBrother(?y, ?z) -> hasUncle(?x, ?z)"
        )

    # 4. Run the Reasoner
    print("[INFO] Running Pellet reasoner...")
    with onto:
        sync_reasoner_pellet(infer_property_values=True)

    # 5. Output the Results
    print("\n=== Reasoner Output ===")
    print(f"[SUCCESS] John's uncle(s) AFTER reasoning: {[u.name for u in john.hasUncle]}")
    print("===================================\n")

if __name__ == "__main__":
    main()