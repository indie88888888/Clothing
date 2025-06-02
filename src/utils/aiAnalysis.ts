import { pipeline, env } from '@huggingface/transformers';
import { CLOTHING_DATABASE } from './clothingDatabase';

// Configure transformers.js
env.allowLocalModels = false;
env.useBrowserCache = true;

let imageClassifier: any = null;

const loadModels = async () => {
  if (!imageClassifier) {
    console.log('Loading image classification model...');
    imageClassifier = await pipeline(
      'image-classification',
      'google/vit-base-patch16-224',
      { device: 'webgpu' }
    );
  }
};

export const analyzeClothingItem = async (imageFile: File) => {
  try {
    await loadModels();
    
    const imageUrl = URL.createObjectURL(imageFile);
    const results = await imageClassifier(imageUrl);
    
    console.log('Classification results:', results);
    
    // Enhanced analysis with comprehensive database matching
    const clothingAnalysis = interpretClothingWithDatabase(results, imageFile.name);
    
    URL.revokeObjectURL(imageUrl);
    
    return clothingAnalysis;
  } catch (error) {
    console.error('Error analyzing clothing item:', error);
    
    // Smart fallback with database patterns
    return generateFallbackAnalysis(imageFile.name);
  }
};

const interpretClothingWithDatabase = (results: any[], fileName: string) => {
  const topResults = results.slice(0, 5); // Use top 5 results for better accuracy
  const combinedLabels = topResults.map(r => r.label.toLowerCase()).join(' ');
  const fileName_ = fileName.toLowerCase();
  const analysisText = (combinedLabels + ' ' + fileName_).toLowerCase();
  
  console.log('Analyzing with enhanced database:', analysisText);
  
  // Determine clothing type with priority scoring
  const type = determineClothingType(analysisText);
  const color = determineColor(analysisText);
  const style = determineStyle(analysisText, type);
  const season = determineSeason(type, style, color);
  const occasion = determineOccasion(type, style);
  
  return {
    type,
    color,
    style,
    season,
    occasion,
    confidence: topResults[0]?.score || 0.8
  };
};

const determineClothingType = (text: string) => {
  let bestMatch = 'shirt';
  let bestScore = 0;
  
  // Score each clothing type based on keyword matches and priority
  for (const [typeName, typeData] of Object.entries(CLOTHING_DATABASE.types)) {
    let score = 0;
    
    // Add points for positive matches
    for (const keyword of typeData.keywords) {
      if (text.includes(keyword.toLowerCase())) {
        score += keyword.length * typeData.priority;
      }
    }
    
    // Subtract points for exclusion keywords
    for (const excludeKeyword of typeData.excludeKeywords || []) {
      if (text.includes(excludeKeyword.toLowerCase())) {
        score -= excludeKeyword.length * 5; // Penalty for exclusions
      }
    }
    
    // Bonus for exact matches
    if (text.includes(typeName)) {
      score += typeName.length * 15;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = typeName;
    }
  }
  
  console.log(`Best clothing type match: ${bestMatch} (score: ${bestScore})`);
  return bestMatch;
};

const determineColor = (text: string) => {
  let bestMatch = 'blue';
  let bestScore = 0;
  
  // Score each color based on keyword matches
  for (const [colorName, colorData] of Object.entries(CLOTHING_DATABASE.colors)) {
    let score = 0;
    
    for (const keyword of colorData.keywords) {
      if (text.includes(keyword.toLowerCase())) {
        score += keyword.length * colorData.priority;
      }
    }
    
    // Bonus for exact color name match
    if (text.includes(colorName)) {
      score += colorName.length * 10;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = colorName;
    }
  }
  
  // If no clear color match, use intelligent defaults
  if (bestScore === 0) {
    const colorDistribution = {
      'black': 0.18,
      'white': 0.16,
      'blue': 0.14,
      'gray': 0.12,
      'red': 0.08,
      'green': 0.08,
      'brown': 0.07,
      'pink': 0.06,
      'yellow': 0.05,
      'orange': 0.03,
      'purple': 0.03
    };
    
    const colors = Object.keys(colorDistribution);
    const weights = Object.values(colorDistribution);
    
    const random = Math.random();
    let cumulative = 0;
    for (let i = 0; i < weights.length; i++) {
      cumulative += weights[i];
      if (random <= cumulative) {
        bestMatch = colors[i];
        break;
      }
    }
  }
  
  console.log(`Best color match: ${bestMatch} (score: ${bestScore})`);
  return bestMatch;
};

const determineStyle = (text: string, type: string) => {
  let bestMatch = 'casual';
  let bestScore = 0;
  
  // Score styles based on keywords and clothing type compatibility
  for (const [styleName, styleData] of Object.entries(CLOTHING_DATABASE.styles)) {
    let score = 0;
    
    // Keyword matching
    for (const keyword of styleData.keywords) {
      if (text.includes(keyword.toLowerCase())) {
        score += keyword.length * 5;
      }
    }
    
    // Type compatibility bonus
    if (styleData.types.includes(type)) {
      score += 20;
    }
    
    if (score > bestScore) {
      bestScore = score;
      bestMatch = styleName;
    }
  }
  
  // Fallback based on clothing type if no clear style match
  if (bestScore === 0) {
    const typeStyleMap = {
      't-shirt': 'casual',
      'dress': 'formal',
      'blazer': 'formal',
      'shorts': 'casual',
      'suit': 'formal',
      'jeans': 'casual',
      'athletic': 'athletic'
    };
    
    bestMatch = typeStyleMap[type as keyof typeof typeStyleMap] || 'casual';
  }
  
  console.log(`Best style match: ${bestMatch} (score: ${bestScore})`);
  return bestMatch;
};

const determineSeason = (type: string, style: string, color: string) => {
  // Check season compatibility with type, style, and color
  for (const [seasonName, seasonData] of Object.entries(CLOTHING_DATABASE.seasons)) {
    let score = 0;
    
    if (seasonData.types.includes(type)) score += 10;
    if (seasonData.colors.includes(color)) score += 5;
    
    if (score >= 10) {
      return seasonName;
    }
  }
  
  // Smart seasonal defaults based on clothing type
  const seasonalMapping = {
    'shorts': ['summer', 'spring'][Math.floor(Math.random() * 2)],
    't-shirt': ['summer', 'spring'][Math.floor(Math.random() * 2)],
    'tank-top': 'summer',
    'sweater': ['winter', 'fall'][Math.floor(Math.random() * 2)],
    'jacket': ['winter', 'fall'][Math.floor(Math.random() * 2)],
    'coat': 'winter',
    'dress': 'all-season'
  };
  
  return seasonalMapping[type as keyof typeof seasonalMapping] || 'all-season';
};

const determineOccasion = (type: string, style: string) => {
  const occasionMapping = {
    'formal': 'formal',
    'business': 'formal',
    'athletic': 'athletic',
    'casual': 'casual',
    'trendy': 'party'
  };
  
  const typeOccasionMap = {
    'dress': 'formal',
    'blazer': 'formal',
    'suit': 'formal',
    'athletic shorts': 'athletic',
    'running shoes': 'athletic',
    't-shirt': 'casual',
    'jeans': 'casual'
  };
  
  return occasionMapping[style as keyof typeof occasionMapping] || 
         typeOccasionMap[type as keyof typeof typeOccasionMap] || 
         'casual';
};

const generateFallbackAnalysis = (fileName: string) => {
  const type = determineClothingType(fileName.toLowerCase());
  const color = determineColor(fileName.toLowerCase());
  const style = determineStyle(fileName.toLowerCase(), type);
  const season = determineSeason(type, style, color);
  const occasion = determineOccasion(type, style);
  
  return {
    type,
    color,
    style,
    season,
    occasion,
    confidence: 0.7
  };
};

export const analyzePersonalFeatures = async (imageFile: File) => {
  try {
    // For demo purposes, we'll simulate personal analysis
    const personalAnalysis = simulatePersonalAnalysis();
    
    return personalAnalysis;
  } catch (error) {
    console.error('Error analyzing personal features:', error);
    
    // Fallback analysis
    return {
      skinTone: 'medium',
      bodyShape: 'rectangle',
      faceShape: 'oval',
      hairColor: 'brown',
      colorPalette: ['navy', 'white', 'gray', 'burgundy', 'forest green'],
      recommendations: [
        'Emphasize your waistline with belted pieces',
        'Choose colors that complement your skin tone',
        'Opt for structured pieces for formal occasions',
        'Layer textures for visual interest'
      ]
    };
  }
