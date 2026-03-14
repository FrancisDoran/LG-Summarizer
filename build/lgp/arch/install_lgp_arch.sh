#!/bin/bash

# Install Link Gram Parser, and the deps for it.

sudo pacman -S --needed base-devel && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si

cd ..

rm -rf yay

yay -S link-grammar

echo "Installation complete...run link-parser"




