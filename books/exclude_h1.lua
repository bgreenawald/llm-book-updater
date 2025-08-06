function Header(el)
  if el.level == 1 then
    return nil  -- Remove level 1 headers from the document
  end
  return el
end
