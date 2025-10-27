#!/bin/bash

# 🚀 Beautiful GPU Monitoring Script with Colors
# Monitors GPU usage with beautiful formatting and colors

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to get color based on utilization
get_util_color() {
    local util=$1
    if [ "$util" -ge 90 ]; then
        echo -e "${RED}${BOLD}"
    elif [ "$util" -ge 70 ]; then
        echo -e "${YELLOW}${BOLD}"
    elif [ "$util" -ge 30 ]; then
        echo -e "${GREEN}${BOLD}"
    else
        echo -e "${CYAN}"
    fi
}

# Function to get color based on temperature
get_temp_color() {
    local temp=$1
    if [ "$temp" -ge 85 ]; then
        echo -e "${RED}${BOLD}"
    elif [ "$temp" -ge 75 ]; then
        echo -e "${YELLOW}${BOLD}"
    else
        echo -e "${GREEN}${BOLD}"
    fi
}

# Function to get color based on memory usage
get_mem_color() {
    local used=$1
    local total=$2
    local percent=$((used * 100 / total))
    if [ "$percent" -ge 90 ]; then
        echo -e "${RED}${BOLD}"
    elif [ "$percent" -ge 70 ]; then
        echo -e "${YELLOW}${BOLD}"
    else
        echo -e "${GREEN}${BOLD}"
    fi
}

# Function to clear screen and show header
show_header() {
    clear
    echo -e "${PURPLE}${BOLD}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}${BOLD}║                    Real-time GPU Performance Dashboard                       ║${NC}"
    echo -e "${PURPLE}${BOLD}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${WHITE}${BOLD}Press Ctrl+C to stop monitoring${NC}"
    echo
}

# Function to move cursor up and clear lines
clear_lines() {
    local lines=$1
    for i in $(seq 1 $lines); do
        echo -ne "\033[1A\033[2K"
    done
}

# Show initial header
show_header

# Main monitoring loop
while true; do
    # Get current timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Count number of GPUs to know how many lines to clear
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    lines_per_gpu=1  # Each GPU display takes 1 line
    total_lines=$((gpu_count * lines_per_gpu + 1))  # +1 for the empty line
    
    # Clear previous GPU output (but keep header)
    clear_lines $total_lines
    
    # Get GPU information and display
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS=',' read -r index name util mem_used mem_total temp power; do
        # Calculate memory percentage
        mem_percent=$((mem_used * 100 / mem_total))
        
        # Get colors
        util_color=$(get_util_color "$util")
        temp_color=$(get_temp_color "$temp")
        mem_color=$(get_mem_color "$mem_used" "$mem_total")
        
        # Create inline progress bars
        util_bar=""
        mem_bar=""
        
        # Utilization bar (10 characters)
        util_filled=$((util / 10))
        for i in $(seq 1 $util_filled); do
            util_bar="${util_bar}█"
        done
        for i in $(seq $((util_filled + 1)) 10); do
            util_bar="${util_bar}░"
        done
        
        # Memory bar (10 characters)
        mem_filled=$((mem_percent / 10))
        for i in $(seq 1 $mem_filled); do
            mem_bar="${mem_bar}█"
        done
        for i in $(seq $((mem_filled + 1)) 10); do
            mem_bar="${mem_bar}░"
        done
        
        # Display GPU information as inline key-value pairs with bars
        echo -e "${WHITE}${BOLD}GPU #${index}${NC} ${CYAN}${name}${NC} | ${YELLOW}Time:${NC} ${timestamp} | ${WHITE}${BOLD}Util:${NC} ${util_color}${util}% [${util_bar}]${NC} | ${WHITE}${BOLD}Mem:${NC} ${mem_color}${mem_used}MB/${mem_total}MB (${mem_percent}%) [${mem_bar}]${NC} | ${WHITE}${BOLD}Temp:${NC} ${temp_color}${temp}°C${NC} | ${WHITE}${BOLD}Power:${NC} ${GREEN}${power}W${NC}"
    done
    
    echo
    sleep 5
done
