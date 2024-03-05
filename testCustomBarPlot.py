import matplotlib.pyplot as plt
import numpy as np

data_dict_list = [
    {
        'Temperature': {
            'label': 'Temperature (°C)',
            'value': 25,
            'max_limit': 30,
            'min_limit': 20,
            'colors': ['blue', 'green', 'red'],  # For example, blue for cold, green for optimal, red for hot
            'decimal': 2
        }
    },
    {
        'Pressure': {
            'label': 'Pressure (atm)',
            'value': 1.05,
            'max_limit': 1.5,
            'min_limit': 0.5,
            'colors': ['lightblue', 'lightgreen', 'lightcoral'],  # For different pressure levels
            'decimal': 2
        }
    },
    {
        'Humidity': {
            'label': 'Humidity (%)',
            'value': 50,
            'max_limit': 70,
            'min_limit': 30,
            'colors': ['skyblue', 'limegreen', 'salmon'],  # For different humidity levels
            'decimal': 2
        }
    }
]

# Here is your function slightly adjusted for correct iteration and indexing:
def create_custom_bar_subplots(data_dict_list):
    num_subplots = len(data_dict_list)
    num_rows = (num_subplots + 1) // 2    
    fig, axs = plt.subplots(num_rows, 2, figsize=(8, 5.5), sharex=False, sharey=True)
    axs = axs.flatten()
    
    for i, data_dict in enumerate(data_dict_list):
        for scalar_name, scalar_data in data_dict.items():
            label = scalar_data['label']
            value = scalar_data['value']
            max_limit = scalar_data['max_limit']
            min_limit = scalar_data['min_limit']
            colors = scalar_data['colors']
            decimal = scalar_data['decimal']
            
            left_bound = min(min_limit - abs(min_limit * 0.05), value - abs(value * 0.05))
            right_bound = max(max_limit + abs(max_limit * 0.05), value + abs(value * 0.05))
            total_width = right_bound - left_bound
    
            # Bar widths in percentage
            left_width = np.round((min_limit - left_bound) / total_width, decimal)
            middle_width = np.round((max_limit - min_limit) / total_width, decimal)
            right_width = np.round((right_bound - max_limit) / total_width, decimal)
            
            bar_height = 0.2 
            axs[i].barh(0, left_width, left=0, color=colors[0], height=bar_height)
            axs[i].barh(0, middle_width, left=left_width, color=colors[1], height=bar_height)
            axs[i].barh(0, right_width, left=left_width+middle_width, color=colors[2], height=bar_height)
            # Add transparent bar to cover the entire width
            # axs[i].barh(0, 1, left=0, color='white', alpha=0)
            
    
            # Place the value indicator
            value_pos = np.round((value - left_bound) / total_width, decimal)
            axs[i].axvline(x=value_pos, color='black', linewidth=4)  # Adjust styling as needed
    
            # Remove y-axis and spines
            axs[i].get_yaxis().set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
    
            # Set limits and labels
            axs[i].set_ylim(-0.5, 0.5)   
            axs[i].set_xlim(0, 1)        
            axs[i].set_xticks([left_width, left_width+middle_width, value_pos])
            axs[i].set_xticklabels([min_limit, max_limit, value])
            axs[i].tick_params(axis='x', which='both', length=0)
            axs[i].set_title(label)
            
    if num_subplots % 2 != 0:
        axs[-1].axis('off')
        
    plt.tight_layout()
    plt.show()

# Now, call the function with the prepared data
create_custom_bar_subplots(data_dict_list)
