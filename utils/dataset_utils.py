def simplify_entity(entity):
    """
    Function used only for the Waiting List corpus, it generalizes entities so as not to have so many classes.
    
    Parameters:
    entity (string): Entity name.
    Returns:
    _ (string): Returns the simplified entity or the original depending on the entity type.
    """
    if entity in ["Laboratory_or_Test_Result", "Sign_or_Symptom", "Clinical_Finding"]:
        return "Finding"
    elif entity in ["Procedure", "Laboratory_Procedure", "Therapeutic_Procedure", "Diagnostic_Procedure"]:
        return "Procedure"
    return entity