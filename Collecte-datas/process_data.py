def process_data(data):
    Nested_values = ['product','ecoscore_data','transportation_values', 'values', 'packaging','threatened_species','agribalyse', 'grades',
                 'previous_data','scores','_keywords','additives_debug_tags','additives_old_tags','additives_prev_original_tags','data_sources_tags','debug_param_sorted_langs',
              'packagings','nutrition']
    skip_values = ['_keywords','additives_debug_tags','additives_old_tags','additives_prev_original_tags','data_sources_tags','debug_param_sorted_langs',
              'packagings','nutrition']

    flattened_data = {}
    for idx,value in enumerate(data):

    #print(idx)
        flattened_data[idx] = {}
        for prop_idx, prop_value in  value.items():

            if prop_idx in Nested_values:
                if prop_idx in skip_values:
                    pass
                else:
                    # loop through each nested property
                    if prop_value is not None:
                        if isinstance(prop_value, dict):
                            for nested_idx, nested_value in prop_value.items():
                                flattened_data[idx][prop_idx+'_'+nested_idx] = prop_value


            else:
                flattened_data[idx][prop_idx] = prop_value
    return flattened_data