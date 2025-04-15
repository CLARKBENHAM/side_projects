// const GEMINI_API_KEY =

function GEMINI_FLASH(prompt) {
  const url = `https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`

  const payload = {
    contents: [
      {
        parts: [{ text: prompt }],
      },
    ],
  }

  const options = {
    method: "post",
    content_type: "application/json",
    payload: JSON.stringify(payload),
  }

  try {
    const response = UrlFetchApp.fetch(url, options)
    const json = JSON.parse(response.getContentText())
    const { candidates } = json
    if (candidates && candidates.length > 0) {
      return candidates[0].content.parts[0].text
    } else {
      return "No response from Gemini."
    }
  } catch (error) {
    return `Error: ${error.message}`
  }
}

/**
 * Makes an API call to Gemini with Google Search grounding enabled
 *
 * @param {string} prompt - The prompt to send to Gemini
 * @return {string} The response from Gemini with web search grounding
 */
function GEMINI_FLASH_SEARCH(prompt, include_citations = False) {
  // const url = 'https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key=' + GEMINI_API_KEY;
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`

  const payload = {
    contents: [
      {
        parts: [{ text: prompt }],
      },
    ],
    // Enable Google Search as a tool
    tools: [
      {
        google_search: {},
      },
    ],
  }

  const options = {
    method: "post",
    content_type: "application/json",
    payload: JSON.stringify(payload),
  }

  try {
    const response = UrlFetchApp.fetch(url, options)
    const json = JSON.parse(response.getContentText())
    const { candidates } = json

    if (candidates && candidates.length > 0) {
      // Extract the model's text response
      const text_response = candidates[0].content.parts[0].text

      // Check if there's grounding metadata
      if (candidates[0].groundingMetadata && include_citations) {
        // Process and add citation information
        return process_grounded_response(
          text_response,
          candidates[0].groundingMetadata
        )
      } else {
        return text_response
      }
    } else {
      return "No response from Gemini."
    }
  } catch (error) {
    return `Error: ${error.message}`
  }
}
/**
 * Process a grounded response to add citation information
 *
 * @param {string} text - The text response from Gemini
 * @param {object} groundingMetadata - The grounding metadata from the response
 * @return {string} The processed response with citations
 */
function process_grounded_response(text, groundingMetadata) {
  // If there's no grounding data, return the original text
  if (
    !grounding_metadata ||
    !groundingMetadata.groundingChunks ||
    !groundingMetadata.groundingSupports
  ) {
    return text
  }

  // Build a simple citation footer
  let result = `${text}\n\n`
  result += "Sources:\n"

  // Add unique sources from grounding chunks
  const sources = new Set()
  groundingMetadata.groundingChunks.forEach((chunk, index) => {
    if (chunk.web && chunk.web.title && chunk.web.uri) {
      sources.add(`${index + 1}. ${chunk.web.title}: ${chunk.web.uri}`)
    }
  })

  // Add sources to the response
  sources.forEach((source) => {
    result += `${source}\n`
  })

  return result
}
